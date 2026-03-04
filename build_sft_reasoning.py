#!/usr/bin/env python3
"""
Convert rewritten DeepSeek responses into SFT format matching sft-all.json,
with reasoning wrapped in <think></think> tags inside the answer.

Collects from specified rewritten response directories. Each question gets
one entry per run included.

Usage:
    python build_sft_reasoning.py                   # all available runs
    python build_sft_reasoning.py --runs 0,1,2,3    # specific runs
"""

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent


def runs_label(run_ids: list[int]) -> str:
    """Generate a compact label like 'runs_0-3' or 'runs_0,1,5,7'."""
    if not run_ids:
        return "runs_none"
    s = sorted(run_ids)
    # Check if consecutive
    if s == list(range(s[0], s[-1] + 1)):
        return f"runs_{s[0]}-{s[-1]}"
    return f"runs_{'_'.join(str(i) for i in s)}"


def build_entry(data: dict) -> dict | None:
    """Build a single SFT entry from a rewritten response file."""
    meta = data.get("_meta", {})
    msg = data["choices"][0]["message"]
    reasoning = msg.get("reasoning_content", "") or ""
    answer = msg.get("content", "") or ""

    if not answer:
        return None

    if reasoning:
        gpt_value = f"<think>\n{reasoning}\n</think>\n\n{answer}"
    else:
        gpt_value = answer

    return {
        "id": meta.get("sft_id", meta.get("idx")),
        "title": meta.get("title", ""),
        "conversations": [
            {"from": "human", "value": meta.get("question", "")},
            {"from": "gpt", "value": gpt_value},
        ],
    }


def find_available_runs() -> list[int]:
    """Find all rewritten_responses_N directories that exist."""
    runs = []
    for d in sorted(BASE_DIR.glob("rewritten_responses_*")):
        try:
            run_id = int(d.name.split("_")[-1])
            runs.append(run_id)
        except ValueError:
            continue
    return runs


def main():
    parser = argparse.ArgumentParser(description="Build SFT reasoning dataset from rewritten responses")
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated run IDs to include (default: all available)")
    args = parser.parse_args()

    available = find_available_runs()
    if args.runs:
        run_ids = [int(x.strip()) for x in args.runs.split(",")]
    else:
        run_ids = available

    label = runs_label(run_ids)
    output_file = BASE_DIR / f"sft-reasoning-{label}.json"

    print(f"Available runs: {available}")
    print(f"Selected runs: {run_ids}")
    print(f"Output: {output_file.name}")

    entries = []
    total_files = 0
    per_run = {}

    for run_id in sorted(run_ids):
        run_dir = BASE_DIR / f"rewritten_responses_{run_id}"
        if not run_dir.exists():
            print(f"  run {run_id}: directory not found, skipping")
            continue

        paths = sorted(run_dir.glob("*.json"), key=lambda p: int(p.stem))
        run_count = 0

        for path in paths:
            with open(path) as f:
                data = json.load(f)

            entry = build_entry(data)
            if entry is None:
                continue

            entries.append(entry)
            run_count += 1
            total_files += 1

        per_run[run_id] = run_count
        print(f"  run {run_id}: {run_count} entries")

    print(f"\nTotal: {total_files} entries from {len(per_run)} runs")

    with open(output_file, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(entries)} entries to {output_file}")


if __name__ == "__main__":
    main()
