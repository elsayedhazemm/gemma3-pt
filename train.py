#!/usr/bin/env python3
"""
Continued PreTraining (CPT) script for Gemma 3 4B on packed medical text.

Loads the pre-tokenized dataset from ./dataset/ and trains with HF Trainer.
See TRAINING_GUIDE.md for launch commands and hyperparameter guidance.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ── Config ──────────────────────────────────────────────────────────────────────

@dataclass
class CPTConfig:
    model_name: str = "google/gemma-3-4b-pt"
    dataset_path: str = str(Path(__file__).parent / "dataset")
    output_dir: str = str(Path(__file__).parent / "checkpoints")

    # Training hyperparameters (tuned for 8x H100 80GB)
    num_train_epochs: int = 8
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1  # effective batch = 8 * 1 * 8 GPUs = 64 seqs = 524K tok/step
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Precision and efficiency
    bf16: bool = True
    tf32: bool = False
    gradient_checkpointing: bool = True
    torch_compile: bool = True  # H100 + PyTorch 2.x = ~15-20% speedup

    # Logging and saving
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 5
    report_to: str = "tensorboard"

    # Eval (optional — split a small portion for perplexity tracking)
    eval_ratio: float = 0.02  # 2% of data for eval
    eval_strategy: str = "steps"
    eval_steps: int = 50

    seed: int = 42


# ── Dataset ─────────────────────────────────────────────────────────────────────

class CausalLMDataset(torch.utils.data.Dataset):
    """Wraps HF Dataset to return input_ids and labels (same as input_ids for CLM)."""

    def __init__(self, hf_dataset: Dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.zeros_like(input_ids),
            "labels": input_ids.clone(),
        }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    cfg = CPTConfig()

    print(f"Loading dataset from {cfg.dataset_path}")
    ds = Dataset.load_from_disk(cfg.dataset_path)
    print(f"Dataset: {ds.shape[0]} sequences, {len(ds[0]['input_ids'])} tokens each")

    # Train/eval split
    if cfg.eval_ratio > 0:
        split = ds.train_test_split(test_size=cfg.eval_ratio, seed=cfg.seed)
        train_ds = CausalLMDataset(split["train"])
        eval_ds = CausalLMDataset(split["test"])
        print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    else:
        train_ds = CausalLMDataset(ds)
        eval_ds = None
        print(f"Train: {len(train_ds)}, Eval: None")

    # Load model
    print(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e9:.2f}B, Trainable: {trainable_params / 1e9:.2f}B")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy if eval_ds else "no",
        eval_steps=cfg.eval_steps if eval_ds else None,
        report_to=cfg.report_to,
        seed=cfg.seed,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        torch_compile=cfg.torch_compile,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_path)
    print(f"Final model saved to {final_path}")

    # Save tokenizer alongside model for easy loading
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.save_pretrained(final_path)
    print(f"Tokenizer saved to {final_path}")


if __name__ == "__main__":
    main()
