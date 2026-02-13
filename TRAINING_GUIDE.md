# Training Guide: CPT for Gemma 3 4B on Medical Text

## Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA
- NVIDIA GPU(s) with bf16 support (Ampere or newer: A100, A10G, H100, RTX 3090/4090)
- The dataset must be built first: `python build_dataset.py`

Install dependencies:
```bash
pip install transformers datasets accelerate torch tensorboard
# For multi-GPU with DeepSpeed:
pip install deepspeed
# For flash attention (highly recommended):
pip install flash-attn --no-build-isolation
```

## Launch Commands

### Single GPU

```bash
python train.py
```

### Multi-GPU (single node)

```bash
torchrun --nproc_per_node=NUM_GPUS train.py
```

For example, on a 4x A100 node:
```bash
torchrun --nproc_per_node=4 train.py
```

### Multi-GPU with DeepSpeed ZeRO Stage 2

```bash
accelerate launch --config_file ds_zero2.yaml train.py
```

Or directly with DeepSpeed:
```bash
deepspeed --num_gpus=4 train.py --deepspeed ds_zero2.json
```

### Multi-Node

```bash
# On each node:
torchrun \
  --nproc_per_node=NUM_GPUS_PER_NODE \
  --nnodes=NUM_NODES \
  --node_rank=NODE_RANK \
  --master_addr=MASTER_IP \
  --master_port=29500 \
  train.py
```

## Recommended Hyperparameters

These are the defaults in `train.py`, tuned for ~50M tokens of domain text on Gemma 3 4B:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning rate** | 2e-5 | Conservative for CPT — we want to inject knowledge without forgetting. 1e-5 to 5e-5 is the typical range. |
| **LR scheduler** | Cosine | Standard for pretraining. Decays smoothly to near-zero. |
| **Warmup** | 5% of steps | Short warmup since we start from a pretrained model. |
| **Epochs** | 3 | Small corpus (~50M tokens) benefits from multiple passes. Monitor eval loss — if it starts rising, you're overfitting. |
| **Batch size** | 2 per device | Fits in 24GB VRAM with gradient checkpointing. |
| **Gradient accumulation** | 8 | Effective batch = 2 × 8 × num_gpus. On 4 GPUs → effective batch of 64 sequences = 524K tokens/step. |
| **Weight decay** | 0.01 | Standard regularization. |
| **Max grad norm** | 1.0 | Gradient clipping for stability. |

### Adjusting for your setup

**If you have more VRAM** (e.g., A100 80GB):
- Increase `per_device_train_batch_size` to 4 or 8
- Decrease `gradient_accumulation_steps` proportionally to maintain the same effective batch size

**If you have less VRAM** (e.g., RTX 3090 24GB):
- Keep `per_device_train_batch_size` at 1 or 2
- Increase `gradient_accumulation_steps` to compensate
- Ensure gradient checkpointing is on

**Effective batch size target**: ~32-128 sequences (262K–1M tokens per step). Larger batches give more stable gradients but fewer update steps.

## Efficiency Tips

### Must-haves
- **bf16**: Always on. Halves memory, no accuracy loss on Ampere+.
- **Gradient checkpointing**: Trades ~30% more compute for ~60% less activation memory. Essential for 4B on 24GB GPUs.
- **Flash Attention 2**: ~2x faster attention, less memory. The train script requests it via `attn_implementation="flash_attention_2"`. Install `flash-attn` separately.

### Nice-to-haves
- **tf32**: Enabled by default on Ampere+. Speeds up matmuls with negligible precision impact.
- **torch.compile**: Can give 10-20% speedup but may have compatibility issues. Set `torch_compile=True` in the config to try it.
- **Dataloader workers**: Set to 4 by default. Increase if you see GPU idle time waiting for data.

### What NOT to do
- Don't use fp16 — Gemma 3 was trained in bf16 and fp16 can cause NaN losses.
- Don't freeze layers — for CPT you want full model updates so the knowledge propagates through all layers.
- Don't use a very high learning rate (>1e-4) — risks catastrophic forgetting of the model's existing capabilities.

## DeepSpeed Configurations

### ZeRO Stage 2 (recommended for multi-GPU)

Save as `ds_zero2.json`:
```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

### ZeRO Stage 3 (if OOM with Stage 2)

Save as `ds_zero3.json`:
```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

To use DeepSpeed, add to the TrainingArguments in `train.py`:
```python
deepspeed="ds_zero2.json",  # or ds_zero3.json
```

## VRAM Estimates

Gemma 3 4B in bf16 ≈ 8GB for model weights alone.

| Setup | per_device_batch | Gradient Checkpointing | Est. VRAM/GPU |
|-------|:---:|:---:|---:|
| 1x RTX 3090 (24GB) | 1 | Yes | ~18-20 GB |
| 1x RTX 3090 (24GB) | 2 | Yes | ~22-24 GB |
| 1x A100 (40GB) | 4 | Yes | ~28-32 GB |
| 1x A100 (80GB) | 8 | Yes | ~45-55 GB |
| 4x A100 (80GB) + ZeRO-2 | 8 | Yes | ~35-45 GB/GPU |

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir checkpoints/
```

Key metrics to watch:
- **train/loss**: Should decrease steadily. If it plateaus early, try increasing learning rate slightly.
- **eval/loss**: Should track train loss. If it rises while train loss drops → overfitting → reduce epochs or increase weight decay.
- **learning_rate**: Verify the cosine schedule looks correct.

### What good CPT training looks like
- Loss decreases quickly in the first ~10% of training (the model is absorbing domain vocabulary and patterns)
- Then decreases more slowly but steadily
- Final loss for medical text CPT on a 4B model is typically 1.5–2.5 (depends on content complexity)
- Eval loss should be close to train loss (within ~0.1-0.2)

### Signs of trouble
- **Loss spikes**: Reduce learning rate or check for data issues
- **NaN loss**: Switch from fp16 to bf16, reduce learning rate
- **Eval loss rising**: Overfitting — stop training or reduce epochs
- **Loss not decreasing**: Learning rate too low, or data too similar to pretraining data (model already knows it)

## After Training

The final model is saved to `checkpoints/final/`. Load it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/final/")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/final/")
```

This model is now domain-adapted to medical text. Common next steps:
1. **SFT (Supervised Fine-Tuning)**: Fine-tune on medical QA pairs for instruction following
2. **Evaluation**: Test on medical benchmarks (MedQA, PubMedQA, etc.)
3. **Merge**: If you used LoRA (not applicable here, we did full CPT), merge adapters
