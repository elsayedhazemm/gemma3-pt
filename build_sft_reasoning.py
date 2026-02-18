#!/usr/bin/env python3
"""
Convert rewritten DeepSeek responses into SFT format matching sft-all.json,
with reasoning wrapped in <think></think> tags inside the answer.

Output: sft-reasoning.json
"""

import json
from pathlib import Path

INPUT_DIR = Path(__file__).parent / "rewritten_responses"
OUTPUT_FILE = Path(__file__).parent / "sft-all-reasoning.json"


def main():
    paths = sorted(INPUT_DIR.glob("*.json"), key=lambda p: int(p.stem))
    print(f"Found {len(paths)} rewritten responses")

    entries = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        meta = data.get("_meta", {})
        msg = data["choices"][0]["message"]
        reasoning = msg.get("reasoning_content", "") or ""
        answer = msg.get("content", "") or ""

        if not answer:
            continue

        # Build the response with <think> tags
        if reasoning:
            gpt_value = f"<think>\n{reasoning}\n</think>\n\n{answer}"
        else:
            gpt_value = answer

        entry = {
            "id": meta.get("sft_id", meta.get("idx")),
            "title": meta.get("title", ""),
            "conversations": [
                {
                    "from": "human",
                    "value": meta.get("question", ""),
                },
                {
                    "from": "gpt",
                    "value": gpt_value,
                },
            ],
        }
        entries.append(entry)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(entries)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
