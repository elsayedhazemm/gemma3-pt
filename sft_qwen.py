#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) script for Qwen3-4B using TRL SFTTrainer.

Expects a HuggingFace Dataset with a "messages" column (built by convert_sft_to_messages.py).
The tokenizer's chat template is applied automatically by SFTTrainer.

Launch with DDP:
  torchrun --nproc_per_node=8 sft_qwen.py
  torchrun --nproc_per_node=8 sft_qwen.py --reasoning   # use reasoning dataset

Single GPU:
  python sft_qwen.py
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


SCRIPT_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Qwen3-4B")
    parser.add_argument(
        "--model-name", type=str, default="base",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to HF Dataset with 'messages' column (default: auto)",
    )
    parser.add_argument(
        "--reasoning", action="store_true",
        help="Use the reasoning dataset (sft_messages_reasoning/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing")
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve dataset path
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    elif args.reasoning:
        dataset_path = SCRIPT_DIR / "sft_messages_reasoning"
    else:
        dataset_path = SCRIPT_DIR / "sft_messages"

    # Resolve output dir
    if args.output_dir:
        output_dir = args.output_dir
    elif args.reasoning:
        output_dir = str(SCRIPT_DIR / "checkpoints_sft_qwen_reasoning")
    else:
        output_dir = str(SCRIPT_DIR / "checkpoints_sft_qwen")

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    ds = Dataset.load_from_disk(str(dataset_path))
    print(f"Dataset: {ds.num_rows} examples")

    # Train/eval split
    eval_ds = None
    if args.eval_ratio > 0:
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
        print(f"Train: {train_ds.num_rows}, Eval: {eval_ds.num_rows}")
    else:
        train_ds = ds
        print(f"Train: {train_ds.num_rows}, Eval: None")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # SFT config
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=args.max_length,
        packing=args.packing,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=50 if eval_ds else None,
        report_to="wandb",
        seed=args.seed,
        dataloader_num_workers=64,
        dataloader_pin_memory=True,
        torch_compile=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # Train
    print("Starting SFT training...")
    trainer.train()

    # Save final model + tokenizer
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model + tokenizer saved to {final_path}")


if __name__ == "__main__":
    main()
