# Training Guide: CPT for Gemma 3 4B on Medical Text

## Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA 12.4+
- 8x NVIDIA H100 80GB (or adjust batch sizes for your setup)
- The dataset must be built first: `python build_dataset.py`

## Requirements

Install with [uv](https://docs.astral.sh/uv/):
```bash
# Create venv and install PyTorch with CUDA 12.4
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core training dependencies
uv pip install transformers datasets accelerate tensorboard

# Flash Attention 2 (required — the train script uses it)
uv pip install flash-attn --no-build-isolation

# Optional: DeepSpeed (not needed for 4B on 8x H100, but useful for larger models)
# uv pip install deepspeed
```

Or install everything at once from the requirements file:
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt --no-build-isolation
```

## Launch Commands

### 8x H100 (recommended)

Plain DDP via `torchrun` — no DeepSpeed needed. The 4B model fits easily in 80GB per GPU.

```bash
torchrun --nproc_per_node=8 train.py
```

With NCCL tuning for optimal H100 NVLink performance:
```bash
NCCL_P2P_LEVEL=NVL \
NCCL_IB_GID_INDEX=3 \
OMP_NUM_THREADS=8 \
torchrun --nproc_per_node=8 train.py
```

### Single GPU (testing / debug)

```bash
python train.py
```

### Multi-GPU (generic)

```bash
torchrun --nproc_per_node=NUM_GPUS train.py
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

## Recommended Hyperparameters (8x H100)

These are the defaults in `train.py`, tuned for 8x H100 80GB with ~35M tokens at 8K context:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning rate** | 2e-5 | Conservative for CPT — injects knowledge without forgetting. 1e-5 to 5e-5 is the typical range. |
| **LR scheduler** | Cosine | Standard for pretraining. Decays smoothly to near-zero. |
| **Warmup** | 5% of steps | Short warmup since we start from a pretrained model. |
| **Epochs** | 8 | Small corpus (~35M tokens) needs many passes for CPT. Pick the best checkpoint by eval loss. |
| **Batch size** | 8 per device | Fits comfortably in H100 80GB (~40-45GB used with gradient checkpointing at 8K context). |
| **Gradient accumulation** | 1 | Effective batch = 8 × 1 × 8 GPUs = 64 sequences = **524K tokens/step**. |
| **Weight decay** | 0.01 | Standard regularization. |
| **Max grad norm** | 1.0 | Gradient clipping for stability. |
| **torch.compile** | True | H100 + PyTorch 2.x gives ~15-20% throughput boost. |

### Training math for this setup

- Dataset: 4,312 packed sequences of 8,192 tokens each
- Effective batch: 64 sequences/step (8 batch × 8 GPUs)
- Steps per epoch: ~67
- Total steps (8 epochs): ~539
- Total token exposure: ~282M tokens (each token seen 8 times)
- Save/eval every 50 steps = ~10 checkpoints per run (keeps best 5)
- Estimated time: ~40-60 minutes total (H100s are fast for a 4B model)

### Adjusting for other setups

**If you have fewer/smaller GPUs** (e.g., 4x A100 40GB):
- Decrease `per_device_train_batch_size` to 2-4
- Increase `gradient_accumulation_steps` to keep effective batch ~64
- Keep gradient checkpointing on

**If you have more VRAM but fewer GPUs** (e.g., 1x H100 80GB):
- Keep `per_device_train_batch_size` at 8
- Increase `gradient_accumulation_steps` to 8 (effective batch = 64)

**Effective batch size target**: ~32-128 sequences (262K–1M tokens per step). Larger batches give more stable gradients but fewer update steps.

## Why DDP (not DeepSpeed) for 4B on 8x H100

The Gemma 3 4B model in bf16 is only ~8GB of weights. Each H100 has 80GB. Even with optimizer states (~16GB), gradients (~8GB), and activations (~12GB at batch=8), you use ~44GB per GPU — well under 80GB.

DeepSpeed ZeRO shards optimizer/gradient/parameter state across GPUs, which **adds communication overhead**. For a model this small on GPUs this large, plain DDP (which only syncs gradients) is faster.

Use DeepSpeed ZeRO-2/3 only if:
- You increase the model size (e.g., Gemma 3 12B or 27B)
- You run out of VRAM with DDP

## Efficiency Tips

### Must-haves
- **bf16**: Always on. Halves memory, no accuracy loss on Ampere+.
- **Gradient checkpointing**: Trades ~30% more compute for ~60% less activation memory. Keeps VRAM headroom for larger batches.
- **Flash Attention 2**: ~2x faster attention, less memory. The train script requests it via `attn_implementation="flash_attention_2"`.
- **torch.compile**: Enabled by default. Fuses kernels for ~15-20% speedup on H100. First step is slow (compilation), then it flies.

### Nice-to-haves
- **tf32**: Enabled by default on Ampere+. Speeds up matmuls with negligible precision impact.
- **Dataloader workers**: Set to 8 to keep the data pipeline ahead of the GPUs.

### What NOT to do
- Don't use fp16 — Gemma 3 was trained in bf16 and fp16 can cause NaN losses.
- Don't freeze layers — for CPT you want full model updates so the knowledge propagates through all layers.
- Don't use a very high learning rate (>1e-4) — risks catastrophic forgetting of the model's existing capabilities.
- Don't use DeepSpeed for a 4B model on 8x H100 — it adds overhead with no benefit.

## DeepSpeed Configurations (for larger models)

Only needed if training a larger model (12B+) or running on smaller GPUs.

### ZeRO Stage 2

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
| 1x RTX 3090 (24GB) | 2 | Yes | ~22-24 GB |
| 1x A100 (40GB) | 4 | Yes | ~28-32 GB |
| 1x A100 (80GB) | 8 | Yes | ~45-55 GB |
| **8x H100 (80GB) DDP** | **8** | **Yes** | **~40-45 GB/GPU** |

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
