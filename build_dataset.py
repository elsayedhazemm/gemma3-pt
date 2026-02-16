#!/usr/bin/env python3
"""
Build a CPT (Continued PreTraining) dataset for Gemma 3 4B from UpToDate medical articles.

Pipeline:
  1. Load all JSON articles from ../clean/
  2. Deduplicate by title (drop exact dupes, keep largest if content differs)
  3. Clean text (strip citations, cross-refs, normalize whitespace)
  4. Format as markdown documents
  5. Shuffle with fixed seed
  6. Tokenize with Gemma 3 tokenizer, pack into 8192-token sequences with EOS separators
  7. Save as HuggingFace Dataset
  8. Write dataset_stats.md
"""

import gc
import json
import re
import random
import statistics
import sys
from pathlib import Path

import pyarrow as pa
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
# ── Config ──────────────────────────────────────────────────────────────────────

CLEAN_DIR = Path(__file__).parent / "clean"
OUTPUT_DIR = Path(__file__).parent / "dataset"
STATS_FILE = Path(__file__).parent / "dataset_stats.md"
MODEL_NAME = "google/gemma-3-4b-pt"
SEQ_LENGTH = 4096
SEED = 42

# ── Step 1: Load articles ──────────────────────────────────────────────────────

def load_articles(clean_dir: Path) -> list[dict]:
    articles = []
    json_files = sorted(clean_dir.glob("*.json"))
    for path in tqdm(json_files, desc="Loading articles"):
        with open(path) as f:
            data = json.load(f)
        data["_filename"] = path.name
        data["_filesize"] = path.stat().st_size
        articles.append(data)
    print(f"Loaded {len(articles)} articles")
    return articles


# ── Step 2: Deduplicate ────────────────────────────────────────────────────────

def deduplicate(articles: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for art in articles:
        title = art.get("title", "")
        groups.setdefault(title, []).append(art)

    deduped = []
    removed = 0
    for title, group in groups.items():
        # Keep the largest file (most content)
        best = max(group, key=lambda a: a["_filesize"])
        deduped.append(best)
        removed += len(group) - 1

    print(f"Deduplicated: {len(articles)} → {len(deduped)} (removed {removed} duplicates)")
    return deduped


# ── Step 3: Clean text ─────────────────────────────────────────────────────────

# Citation markers: [1], [2,3], [1-5], [2,3,5-7], etc.
RE_CITATIONS = re.compile(r"\s*\[\d+[\d,\-\s]*\]")

# Cross-references: (See "Topic name".) and (See 'Topic name'.)
# Also handles multiple see-refs and trailing periods inside parens
RE_CROSSREFS = re.compile(
    r"\s*\(See\s+[\"'][^\"']*[\"'][^)]*\.\)",
    re.IGNORECASE,
)

# Standalone "(See ... above.)" or "(See ... below.)" internal refs
RE_INTERNAL_REFS = re.compile(
    r"\s*\(See\s+'[^']*'\s+(?:above|below)\.?\)",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = RE_CITATIONS.sub("", text)
    text = RE_CROSSREFS.sub("", text)
    text = RE_INTERNAL_REFS.sub("", text)
    # Normalize non-breaking spaces
    text = text.replace("\xa0", " ")
    # Collapse runs of 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ── Step 4: Format as markdown ─────────────────────────────────────────────────

def format_article(article: dict) -> str:
    title = article.get("title", "Untitled")
    sections = [
        (k, v) for k, v in article.items()
        if k not in ("title", "_filename", "_filesize") and isinstance(v, str)
    ]

    parts = [f"# {title}"]
    for section_name, section_text in sections:
        cleaned = clean_text(section_text)
        if cleaned:
            parts.append(f"## {section_name}\n\n{cleaned}")

    return "\n\n".join(parts)


# ── Step 5-6: Tokenize + Pack ──────────────────────────────────────────────────

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

        # Drain buffer into pending sequences
        while len(buffer) >= seq_length:
            pending.append(buffer[:seq_length])
            buffer = buffer[seq_length:]
            num_sequences += 1

            # Flush pending batch to disk
            if len(pending) >= WRITE_BATCH:
                batch = pa.record_batch(
                    [pa.array(pending, type=pa.list_(pa.int32()))],
                    schema=schema,
                )
                writer.write_batch(batch)
                pending.clear()

    # Flush remaining sequences
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


# ── Step 7: Write stats ────────────────────────────────────────────────────────

def write_stats(
    total_articles: int,
    after_dedup: int,
    doc_token_lengths: list[int],
    num_sequences: int,
    seq_length: int,
    stats_path: Path,
):
    total_tokens = sum(doc_token_lengths)
    mean_tok = statistics.mean(doc_token_lengths)
    median_tok = statistics.median(doc_token_lengths)
    stdev_tok = statistics.stdev(doc_token_lengths) if len(doc_token_lengths) > 1 else 0

    # Token length distribution buckets
    buckets = [
        ("<1K", 0, 1000),
        ("1–2K", 1000, 2000),
        ("2–4K", 2000, 4000),
        ("4–8K", 4000, 8000),
        ("8–16K", 8000, 16000),
        ("16–32K", 16000, 32000),
        (">32K", 32000, float("inf")),
    ]
    bucket_counts = {label: 0 for label, _, _ in buckets}
    for t in doc_token_lengths:
        for label, lo, hi in buckets:
            if lo <= t < hi:
                bucket_counts[label] += 1
                break

    lines = [
        "# CPT Dataset Statistics",
        "",
        "## Source",
        f"- **Model**: `{MODEL_NAME}`",
        "- **Source directory**: `./clean/`",
        f"- **Sequence length**: {seq_length:,}",
        f"- **Seed**: {SEED}",
        "",
        "## Articles",
        f"- **Total raw articles**: {total_articles:,}",
        f"- **After deduplication**: {after_dedup:,}",
        f"- **Removed duplicates**: {total_articles - after_dedup:,}",
        "",
        "## Token Counts (per article, Gemma 3 tokenizer)",
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
    articles = load_articles(CLEAN_DIR.resolve())

    # Dedup
    total_raw = len(articles)
    articles = deduplicate(articles)

    # Format
    print("Formatting articles...")
    documents = [format_article(art) for art in tqdm(articles, desc="Formatting")]

    # Shuffle
    random.seed(SEED)
    random.shuffle(documents)
    print(f"Shuffled {len(documents)} documents (seed={SEED})")

    # Tokenize + pack → writes Arrow file directly to disk (no big Python list)
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrow_path = OUTPUT_DIR / "_packed.arrow"
    num_sequences, doc_token_lengths = tokenize_and_pack(
        documents, tokenizer, SEQ_LENGTH, arrow_path,
    )

    del documents
    gc.collect()

    # Load the Arrow file memory-mapped (barely uses RAM) and save as HF Dataset
    print("Converting Arrow file to HuggingFace Dataset...")
    source = pa.memory_map(str(arrow_path), "r")
    table = pa.ipc.open_file(source).read_all()
    ds = Dataset(table)
    ds.save_to_disk(str(OUTPUT_DIR.resolve()))
    del ds, table, source
    gc.collect()

    # Clean up temp Arrow file
    arrow_path.unlink(missing_ok=True)
    print(f"Dataset saved to {OUTPUT_DIR.resolve()}")

    # Stats
    write_stats(
        total_articles=total_raw,
        after_dedup=len(articles),
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
    vocab_size = tokenizer.vocab_size
    errors = []

    # 1. Shape
    print(f"Shape: {reloaded.shape}")
    print(f"Columns: {reloaded.column_names}")
    if reloaded.num_rows != num_sequences:
        errors.append(f"Row count mismatch: expected {num_sequences}, got {reloaded.num_rows}")
    if "input_ids" not in reloaded.column_names:
        errors.append("Missing 'input_ids' column")

    # 2. Sequence lengths
    print("Checking all sequence lengths...")
    bad_lengths = 0
    for i in range(reloaded.num_rows):
        if len(reloaded[i]["input_ids"]) != SEQ_LENGTH:
            bad_lengths += 1
    if bad_lengths:
        errors.append(f"{bad_lengths} sequences have wrong length (expected {SEQ_LENGTH})")
    else:
        print(f"  All {reloaded.num_rows} sequences have length {SEQ_LENGTH}")

    # 3. Token ID range
    print("Checking token ID ranges...")
    sample_indices = random.sample(range(reloaded.num_rows), min(200, reloaded.num_rows))
    for idx in sample_indices:
        ids = reloaded[idx]["input_ids"]
        mn, mx = min(ids), max(ids)
        if mn < 0 or mx >= vocab_size:
            errors.append(f"Sequence {idx}: token IDs out of range [{mn}, {mx}] (vocab_size={vocab_size})")
    print(f"  Sampled {len(sample_indices)} sequences — all IDs in [0, {vocab_size})")

    # 4. Decode sanity check
    print("Decoding 3 random sequences...")
    random.seed(SEED)
    check_indices = random.sample(range(reloaded.num_rows), min(3, reloaded.num_rows))
    for idx in check_indices:
        text = tokenizer.decode(reloaded[idx]["input_ids"], skip_special_tokens=False)
        print(f"\n--- Sequence {idx} (first 500 chars) ---")
        print(text[:500])
        print("...")

    # Final verdict
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
