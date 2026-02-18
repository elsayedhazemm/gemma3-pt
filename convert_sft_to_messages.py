#!/usr/bin/env python3
"""
Convert ShareGPT-style SFT datasets (sft-all.json, sft-all-reasoning.json)
into HuggingFace Datasets with a "messages" column suitable for TRL SFTTrainer
with Qwen3's chat template.

Input format (ShareGPT):
  [{"id": ..., "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]

Output format (HF messages):
  Dataset with column "messages":
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

Usage:
  python convert_sft_to_messages.py                          # converts both files
  python convert_sft_to_messages.py --input sft-all.json     # convert specific file
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}

SCRIPT_DIR = Path(__file__).parent


def convert_entry(entry: dict) -> dict | None:
    """Convert a single ShareGPT entry to HF messages format."""
    messages = []
    for turn in entry["conversations"]:
        role = ROLE_MAP.get(turn["from"])
        if role is None:
            print(f"  WARNING: unknown role '{turn['from']}' in entry {entry.get('id')}, skipping entry")
            return None
        content = turn["value"].strip()
        if not content:
            print(f"  WARNING: empty content for role '{role}' in entry {entry.get('id')}, skipping entry")
            return None
        messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None

    return {"messages": messages}


def convert_file(input_path: Path, output_dir: Path):
    """Convert a ShareGPT JSON file to an HF Dataset saved to disk."""
    print(f"Loading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)
    print(f"  {len(data)} entries loaded")

    converted = []
    skipped = 0
    for entry in data:
        result = convert_entry(entry)
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    print(f"  {len(converted)} entries converted, {skipped} skipped")

    ds = Dataset.from_list(converted)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"  Saved to {output_dir} ({ds.num_rows} rows)")

    # Print a sample to verify
    sample = ds[0]["messages"]
    print(f"\n  Sample entry ({len(sample)} turns):")
    for msg in sample:
        preview = msg["content"][:120].replace("\n", "\\n")
        print(f"    [{msg['role']}] {preview}...")

    return ds


def main():
    parser = argparse.ArgumentParser(description="Convert ShareGPT SFT data to HF messages format")
    parser.add_argument("--input", type=str, default=None, help="Specific input JSON file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (auto-derived if not set)")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = SCRIPT_DIR / f"sft_messages_{input_path.stem}"
        convert_file(input_path, output_dir)
    else:
        # Convert both default files
        defaults = [
            (SCRIPT_DIR / "sft-all.json", SCRIPT_DIR / "sft_messages"),
            (SCRIPT_DIR / "sft-all-reasoning.json", SCRIPT_DIR / "sft_messages_reasoning"),
        ]
        for input_path, output_dir in defaults:
            if input_path.exists():
                convert_file(input_path, output_dir)
            else:
                print(f"Skipping {input_path} (not found)")


if __name__ == "__main__":
    main()
