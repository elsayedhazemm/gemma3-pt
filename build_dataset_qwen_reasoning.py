#!/usr/bin/env python3
"""
Build a CPT dataset for Qwen3-4B-Base from Claude reasoning traces.

Pipeline:
  1. Load reasoning traces from sft-paired-with-reasoning.json
  2. Extract claude_reasoning field from each entry
  3. Shuffle with fixed seed
  4. Tokenize with Qwen3 tokenizer, pack into 4096-token sequences with EOS separators
  5. Save as HuggingFace Dataset
  6. Write dataset_stats_qwen_reasoning.md
"""

import gc
import json
import random
import statistics
import sys
from pathlib import Path

import pyarrow as pa
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────────

SRC_FILE = Path(__file__).parent / "sft-paired-with-reasoning.json"
OUTPUT_DIR = Path(__file__).parent / "dataset_qwen_reasoning"
STATS_FILE = Path(__file__).parent / "dataset_stats_qwen_reasoning.md"
MODEL_NAME = "Qwen/Qwen3-4B-Base"
SEQ_LENGTH = 4096
SEED = 42

# ── Step 1: Load reasoning traces ────────────────────────────────────────────

def load_reasoning_traces(src_file: Path) -> list[str]:
    with open(src_file) as f:
        data = json.load(f)
    documents = []
    skipped = 0
    for entry in data:
        text = entry.get("claude_reasoning", "").strip()
        if text:
            documents.append(text)
        else:
            skipped += 1
    print(f"Loaded {len(documents)} reasoning traces ({skipped} empty, skipped)")
    return documents


# ── Step 2: Tokenize + Pack ──────────────────────────────────────────────────

def tokenize_and_pack(
    documents: list[str],
    tokenizer,
    seq_length: int,
    output_path: Path,
) -> tuple[int, list[int]]:
    """Tokenize, pack into fixed-length sequences, and write directly to an
    Arrow IPC file on disk.  Only a small buffer is ever held in memory."""
    eos_id = tokenizer.eos_token_id
    schema = pa.schema([("input_ids", pa.list_(pa.int32()))])

    print("Tokenizing, packing, and writing to disk (streaming)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pa.ipc.new_file(str(output_path), schema)

    buffer: list[int] = []
    pending: list[list[int]] = []
    doc_token_lengths: list[int] = []
    total_tokens = 0
    num_sequences = 0
    WRITE_BATCH = 256

    for doc in tqdm(documents, desc="Tokenizing & packing"):
        ids = tokenizer.encode(doc, add_special_tokens=False)
        doc_token_lengths.append(len(ids))
        total_tokens += len(ids) + 1  # +1 for EOS

        buffer.extend(ids)
        buffer.append(eos_id)

        while len(buffer) >= seq_length:
            pending.append(buffer[:seq_length])
            buffer = buffer[seq_length:]
            num_sequences += 1

            if len(pending) >= WRITE_BATCH:
                batch = pa.record_batch(
                    [pa.array(pending, type=pa.list_(pa.int32()))],
                    schema=schema,
                )
                writer.write_batch(batch)
                pending.clear()

    if pending:
        batch = pa.record_batch(
            [pa.array(pending, type=pa.list_(pa.int32()))],
            schema=schema,
        )
        writer.write_batch(batch)
        pending.clear()

    writer.close()

    discarded = len(buffer)
    print(f"Total tokens (with EOS separators): {total_tokens:,}")
    print(f"Packed into {num_sequences:,} sequences of {seq_length} tokens")
    print(f"Discarded {discarded:,} trailing tokens")

    return num_sequences, doc_token_lengths


# ── Step 3: Write stats ──────────────────────────────────────────────────────

def write_stats(
    total_docs: int,
    doc_token_lengths: list[int],
    num_sequences: int,
    seq_length: int,
    stats_path: Path,
):
    total_tokens = sum(doc_token_lengths)
    mean_tok = statistics.mean(doc_token_lengths)
    median_tok = statistics.median(doc_token_lengths)
    stdev_tok = statistics.stdev(doc_token_lengths) if len(doc_token_lengths) > 1 else 0

    buckets = [
        ("<256", 0, 256),
        ("256–512", 256, 512),
        ("512–1K", 512, 1000),
        ("1–2K", 1000, 2000),
        ("2–4K", 2000, 4000),
        (">4K", 4000, float("inf")),
    ]
    bucket_counts = {label: 0 for label, _, _ in buckets}
    for t in doc_token_lengths:
        for label, lo, hi in buckets:
            if lo <= t < hi:
                bucket_counts[label] += 1
                break

    lines = [
        "# CPT Dataset Statistics — Reasoning Traces (Qwen3-4B-Base)",
        "",
        "## Source",
        f"- **Model**: `{MODEL_NAME}`",
        f"- **Source file**: `{SRC_FILE.name}`",
        f"- **Field**: `claude_reasoning`",
        f"- **Sequence length**: {seq_length:,}",
        f"- **Seed**: {SEED}",
        "",
        "## Documents",
        f"- **Total reasoning traces**: {total_docs:,}",
        "",
        "## Token Counts (per document, Qwen3 tokenizer)",
        f"- **Total tokens**: {total_tokens:,}",
        f"- **Mean**: {mean_tok:,.0f}",
        f"- **Median**: {median_tok:,.0f}",
        f"- **Stdev**: {stdev_tok:,.0f}",
        f"- **Min**: {min(doc_token_lengths):,}",
        f"- **Max**: {max(doc_token_lengths):,}",
        "",
        "## Token Length Distribution",
        "| Bucket | Count | Percent |",
        "|--------|------:|--------:|",
    ]
    for label, _, _ in buckets:
        count = bucket_counts[label]
        pct = count / len(doc_token_lengths) * 100
        lines.append(f"| {label} | {count:,} | {pct:.1f}% |")

    lines += [
        "",
        "## Packed Sequences",
        f"- **Total sequences**: {num_sequences:,}",
        f"- **Sequence length**: {seq_length:,} tokens",
        f"- **Total training tokens**: {num_sequences * seq_length:,}",
        "",
    ]

    content = "\n".join(lines)
    stats_path.write_text(content)
    print(f"\nStats written to {stats_path}")
    print(content)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    # Load
    documents = load_reasoning_traces(SRC_FILE.resolve())

    # Shuffle
    random.seed(SEED)
    random.shuffle(documents)
    print(f"Shuffled {len(documents)} documents (seed={SEED})")

    # Tokenize + pack
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrow_path = OUTPUT_DIR / "_packed.arrow"
    num_sequences, doc_token_lengths = tokenize_and_pack(
        documents, tokenizer, SEQ_LENGTH, arrow_path,
    )

    total_docs = len(documents)
    del documents
    gc.collect()

    # Convert Arrow → HF Dataset
    print("Converting Arrow file to HuggingFace Dataset...")
    source = pa.memory_map(str(arrow_path), "r")
    table = pa.ipc.open_file(source).read_all()
    ds = Dataset(table)
    ds.save_to_disk(str(OUTPUT_DIR.resolve()))
    del ds, table, source
    gc.collect()

    arrow_path.unlink(missing_ok=True)
    print(f"Dataset saved to {OUTPUT_DIR.resolve()}")

    # Stats
    write_stats(
        total_docs=total_docs,
        doc_token_lengths=doc_token_lengths,
        num_sequences=num_sequences,
        seq_length=SEQ_LENGTH,
        stats_path=STATS_FILE.resolve(),
    )

    # ── Verification ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATASET INTEGRITY CHECK")
    print("=" * 60)
    reloaded = Dataset.load_from_disk(str(OUTPUT_DIR.resolve()))
    vocab_size = len(tokenizer)
    errors = []

    print(f"Shape: {reloaded.shape}")
    print(f"Columns: {reloaded.column_names}")
    if reloaded.num_rows != num_sequences:
        errors.append(f"Row count mismatch: expected {num_sequences}, got {reloaded.num_rows}")
    if "input_ids" not in reloaded.column_names:
        errors.append("Missing 'input_ids' column")

    print("Checking all sequence lengths...")
    bad_lengths = 0
    for i in range(reloaded.num_rows):
        if len(reloaded[i]["input_ids"]) != SEQ_LENGTH:
            bad_lengths += 1
    if bad_lengths:
        errors.append(f"{bad_lengths} sequences have wrong length (expected {SEQ_LENGTH})")
    else:
        print(f"  All {reloaded.num_rows} sequences have length {SEQ_LENGTH}")

    print("Checking token ID ranges...")
    sample_indices = random.sample(range(reloaded.num_rows), min(200, reloaded.num_rows))
    for idx in sample_indices:
        ids = reloaded[idx]["input_ids"]
        mn, mx = min(ids), max(ids)
        if mn < 0 or mx >= vocab_size:
            errors.append(f"Sequence {idx}: token IDs out of range [{mn}, {mx}] (vocab_size={vocab_size})")
    print(f"  Sampled {len(sample_indices)} sequences — all IDs in [0, {vocab_size})")

    print("Decoding 3 random sequences...")
    random.seed(SEED)
    check_indices = random.sample(range(reloaded.num_rows), min(3, reloaded.num_rows))
    for idx in check_indices:
        text = tokenizer.decode(reloaded[idx]["input_ids"], skip_special_tokens=False)
        print(f"\n--- Sequence {idx} (first 500 chars) ---")
        print(text[:500])
        print("...")

    print("\n" + "=" * 60)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print("DATASET VERIFICATION FAILED")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED — dataset is ready for training")
    print("=" * 60)


if __name__ == "__main__":
    main()
