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

import json
import os
import re
import random
import statistics
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────────

CLEAN_DIR = Path(__file__).parent / ".." / "clean"
OUTPUT_DIR = Path(__file__).parent / "dataset"
STATS_FILE = Path(__file__).parent / "dataset_stats.md"
MODEL_NAME = "google/gemma-3-4b-pt"
SEQ_LENGTH = 8192
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
) -> list[list[int]]:
    eos_id = tokenizer.eos_token_id

    # Tokenize all documents
    print("Tokenizing documents...")
    all_token_ids = []
    doc_token_lengths = []
    for doc in tqdm(documents, desc="Tokenizing"):
        ids = tokenizer.encode(doc, add_special_tokens=False)
        doc_token_lengths.append(len(ids))
        all_token_ids.extend(ids)
        all_token_ids.append(eos_id)

    total_tokens = len(all_token_ids)
    print(f"Total tokens (with EOS separators): {total_tokens:,}")

    # Pack into fixed-length sequences
    num_sequences = total_tokens // seq_length
    packed = []
    for i in range(num_sequences):
        start = i * seq_length
        packed.append(all_token_ids[start : start + seq_length])

    discarded = total_tokens - (num_sequences * seq_length)
    print(f"Packed into {num_sequences:,} sequences of {seq_length} tokens")
    print(f"Discarded {discarded:,} trailing tokens")

    return packed, doc_token_lengths


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
        f"- **Source directory**: `../clean/`",
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

    # Tokenize + pack
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    packed, doc_token_lengths = tokenize_and_pack(documents, tokenizer, SEQ_LENGTH)

    # Build HF Dataset
    print("Building HuggingFace Dataset...")
    ds = Dataset.from_dict({
        "input_ids": packed,
        "attention_mask": [[1] * SEQ_LENGTH for _ in packed],
    })

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR.resolve()))
    print(f"Dataset saved to {OUTPUT_DIR.resolve()}")

    # Stats
    write_stats(
        total_articles=total_raw,
        after_dedup=len(articles),
        doc_token_lengths=doc_token_lengths,
        num_sequences=len(packed),
        seq_length=SEQ_LENGTH,
        stats_path=STATS_FILE.resolve(),
    )

    # Sanity check: decode a few random sequences
    print("\n" + "=" * 60)
    print("SANITY CHECK: Decoding 3 random sequences")
    print("=" * 60)
    random.seed(SEED)
    check_indices = random.sample(range(len(packed)), min(3, len(packed)))
    for idx in check_indices:
        text = tokenizer.decode(packed[idx], skip_special_tokens=False)
        print(f"\n--- Sequence {idx} (first 500 chars) ---")
        print(text[:500])
        print("...")

    # Verify reload
    print("\n" + "=" * 60)
    print("VERIFICATION: Reloading dataset")
    print("=" * 60)
    reloaded = Dataset.load_from_disk(str(OUTPUT_DIR.resolve()))
    print(f"Shape: {reloaded.shape}")
    print(f"Columns: {reloaded.column_names}")
    print(f"First row input_ids length: {len(reloaded[0]['input_ids'])}")
    print("\nDone!")


if __name__ == "__main__":
    main()
