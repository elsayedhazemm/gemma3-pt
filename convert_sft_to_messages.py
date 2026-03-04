#!/usr/bin/env python3
"""
Convert ShareGPT-style SFT datasets into HuggingFace Datasets with a
"messages" column suitable for TRL SFTTrainer.

Input format (ShareGPT):
  [{"id": ..., "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]

Output format (HF messages):
  Dataset with column "messages":
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

Usage:
  python convert_sft_to_messages.py                          # converts default files
  python convert_sft_to_messages.py --input sft-all.json     # convert specific file
  python convert_sft_to_messages.py --runs 0,1,2,3           # build + convert specific runs
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

    if not converted:
        print("  No entries to convert, skipping.")
        return None

    ds = Dataset.from_list(converted).shuffle(seed=42)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"  Saved to {output_dir} ({ds.num_rows} rows, shuffled)")

    # Print a sample to verify
    sample = ds[0]["messages"]
    print(f"\n  Sample entry ({len(sample)} turns):")
    for msg in sample:
        preview = msg["content"][:120].replace("\n", "\\n")
        print(f"    [{msg['role']}] {preview}...")

    return ds


def runs_label(run_ids: list[int]) -> str:
    """Generate a compact label like 'runs_0-3' or 'runs_0_1_5_7'."""
    s = sorted(run_ids)
    if s == list(range(s[0], s[-1] + 1)):
        return f"runs_{s[0]}-{s[-1]}"
    return f"runs_{'_'.join(str(i) for i in s)}"


def find_available_runs() -> list[int]:
    """Find all rewritten_responses_N directories that exist."""
    runs = []
    for d in sorted(SCRIPT_DIR.glob("rewritten_responses_*")):
        try:
            run_id = int(d.name.split("_")[-1])
            runs.append(run_id)
        except ValueError:
            continue
    return runs


def main():
    parser = argparse.ArgumentParser(description="Convert ShareGPT SFT data to HF messages format")
    parser.add_argument("--input", type=str, default=None, help="Specific input JSON file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (auto-derived if not set)")
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated run IDs — builds reasoning dataset from these runs then converts")
    args = parser.parse_args()

    if args.runs:
        # Build reasoning dataset for specified runs, then convert it
        import subprocess
        run_ids = [int(x.strip()) for x in args.runs.split(",")]
        label = runs_label(run_ids)
        sft_file = SCRIPT_DIR / f"sft-reasoning-{label}.json"
        out_dir = SCRIPT_DIR / f"sft_messages_reasoning_{label}"

        print(f"Building SFT reasoning for {label}...")
        subprocess.run(
            ["python", str(SCRIPT_DIR / "build_sft_reasoning.py"), "--runs", args.runs],
            check=True,
        )
        print()
        convert_file(sft_file, out_dir)

    elif args.input:
        input_path = Path(args.input)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = SCRIPT_DIR / f"sft_messages_{input_path.stem}"
        convert_file(input_path, output_dir)

    else:
        # Convert default files + auto-detect reasoning runs
        defaults = [
            (SCRIPT_DIR / "sft-all.json", SCRIPT_DIR / "sft_messages"),
            (SCRIPT_DIR / "sft-all-reasoning.json", SCRIPT_DIR / "sft_messages_reasoning"),
        ]
        for input_path, output_dir in defaults:
            if input_path.exists():
                convert_file(input_path, output_dir)
            else:
                print(f"Skipping {input_path} (not found)")

        # Auto-detect and convert reasoning run files
        available = find_available_runs()
        if available:
            label = runs_label(available)
            sft_file = SCRIPT_DIR / f"sft-reasoning-{label}.json"
            if sft_file.exists():
                out_dir = SCRIPT_DIR / f"sft_messages_reasoning_{label}"
                convert_file(sft_file, out_dir)


if __name__ == "__main__":
    main()
