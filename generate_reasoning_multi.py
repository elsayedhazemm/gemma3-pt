#!/usr/bin/env python3
"""
Generate additional answer variants per question using DeepSeek Reasoner.

Automatically spawns multiple worker processes (default 8) to maximise
throughput. Each worker runs its own asyncio event loop with its own
connection pool.

Interrupt-safe: each run writes to its own directory and skips already-
completed questions on restart.

Usage:
    python generate_reasoning_multi.py --runs 4,5,6         # specific runs
    python generate_reasoning_multi.py --runs 4              # one run, still 8 workers
    python generate_reasoning_multi.py --runs 4 --workers 4  # fewer workers
    python generate_reasoning_multi.py --combine             # only combine existing runs
"""

import argparse
import asyncio
import json
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

from openai import AsyncOpenAI, APIStatusError, APITimeoutError, APIConnectionError
from tqdm.asyncio import tqdm_asyncio

# ── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
SFT_FILE = BASE_DIR / "sft-all.json"
CLEAN_DIR = BASE_DIR / "clean"
TITLES_FILE = BASE_DIR / "titles.json"

ORIGINAL_DIR = BASE_DIR / "deepseek_responses"       # run 0 (already done)
MULTI_DIR = BASE_DIR / "deepseek_responses_multi"     # runs 1-9 go here
COMBINED_DIR = BASE_DIR / "deepseek_responses_10x"    # final combined output

MODEL = "deepseek-reasoner"
MAX_CONCURRENT = 64
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2
NUM_NEW_RUNS = 9  # runs 1 through 9
DEFAULT_WORKERS = 8

SYSTEM_PROMPT = (
    "Answer the following question comprehensively and accurately, "
    "using the provided article to inform your response."
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def build_title_to_file_map() -> dict[str, Path]:
    with open(TITLES_FILE) as f:
        titles = json.load(f)
    title_to_path: dict[str, Path] = {}
    for num, title in titles.items():
        path = CLEAN_DIR / f"{num}.json"
        if path.exists() and title not in title_to_path:
            title_to_path[title] = path
    return title_to_path


def load_article_text(path: Path) -> str:
    with open(path) as f:
        data = json.load(f)
    title = data.get("title", "Untitled")
    parts = [f"# {title}"]
    for key, value in data.items():
        if key != "title" and isinstance(value, str) and value.strip():
            parts.append(f"## {key}\n\n{value}")
    return "\n\n".join(parts)


def load_valid_entries() -> list[tuple[int, dict, Path]]:
    with open(SFT_FILE) as f:
        entries = json.load(f)
    title_to_path = build_title_to_file_map()
    valid = []
    for idx, entry in enumerate(entries):
        path = title_to_path.get(entry["title"])
        if path:
            valid.append((idx, entry, path))
    return valid


# ── API call ─────────────────────────────────────────────────────────────────


async def process_question(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    idx: int,
    entry: dict,
    article_path: Path,
    output_dir: Path,
    run_id: int,
) -> dict:
    entry_id = entry["id"]
    title = entry["title"]
    question = next(
        c["value"] for c in entry["conversations"] if c["from"] == "human"
    )

    article_text = load_article_text(article_path)
    user_content = f"### Article\n\n{article_text}\n\n### Question\n\n{question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_error = None
    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )

                raw = response.model_dump()
                raw["_meta"] = {
                    "idx": idx,
                    "sft_id": entry_id,
                    "title": title,
                    "question": question,
                    "run_id": run_id,
                }

                output_path = output_dir / f"{idx}.json"
                with open(output_path, "w") as f:
                    json.dump(raw, f, indent=2, ensure_ascii=False)

                tokens = raw.get("usage", {}).get("total_tokens", 0)
                return {"idx": idx, "status": "ok", "tokens": tokens}

            except (APIStatusError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                if isinstance(e, APIStatusError) and e.status_code < 500 and e.status_code not in (429, 408):
                    return {"idx": idx, "status": "error", "error": str(e)}
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)

            except Exception as e:
                return {"idx": idx, "status": "error", "error": str(e)}

    return {"idx": idx, "status": "error", "error": f"failed after {MAX_RETRIES} retries: {last_error}"}


# ── Worker process ──────────────────────────────────────────────────────────


def worker_main(
    worker_id: int,
    run_id: int,
    chunk: list[tuple[int, dict, Path]],
    api_key: str,
    concurrency: int,
):
    """Entry point for each worker process. Runs its own asyncio loop."""
    async def _run():
        run_dir = MULTI_DIR / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Re-check what's done (other workers may have written files too)
        done = {int(p.stem) for p in run_dir.glob("*.json")}
        remaining = [(idx, e, p) for idx, e, p in chunk if idx not in done]

        if not remaining:
            print(f"  [worker {worker_id}] nothing to do")
            return

        print(f"  [worker {worker_id}] processing {len(remaining)} questions (concurrency={concurrency})")

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        semaphore = asyncio.Semaphore(concurrency)

        tasks = [
            process_question(client, semaphore, idx, entry, path, run_dir, run_id)
            for idx, entry, path in remaining
        ]

        t0 = time.time()
        results = await tqdm_asyncio.gather(
            *tasks, desc=f"W{worker_id}/R{run_id}", position=worker_id
        )
        elapsed = time.time() - t0

        ok = sum(1 for r in results if r["status"] == "ok")
        errors = [r for r in results if r["status"] == "error"]
        total_tokens = sum(r.get("tokens", 0) for r in results if r["status"] == "ok")

        print(f"  [worker {worker_id}] done in {elapsed:.0f}s — ok: {ok}, errors: {len(errors)}, tokens: {total_tokens:,}")
        if errors:
            for r in errors[:3]:
                print(f"    idx {r['idx']}: {r['error']}")

    asyncio.run(_run())


# ── Run with multiple workers ───────────────────────────────────────────────


def run_with_workers(
    run_id: int,
    valid_entries: list[tuple[int, dict, Path]],
    api_key: str,
    num_workers: int,
    concurrency: int,
    limit: int = 0,
):
    run_dir = MULTI_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    done = {int(p.stem) for p in run_dir.glob("*.json")}
    remaining = [(idx, e, p) for idx, e, p in valid_entries if idx not in done]

    if limit > 0:
        remaining = remaining[:limit]

    print(f"\n{'═' * 60}")
    print(f"  Run {run_id} — done: {len(done)}, remaining: {len(remaining)}, workers: {num_workers}")
    print(f"{'═' * 60}")

    if not remaining:
        print(f"  Run {run_id} already complete, skipping.")
        return

    # Split remaining into chunks for each worker
    chunks: list[list] = [[] for _ in range(num_workers)]
    for i, item in enumerate(remaining):
        chunks[i % num_workers].append(item)

    # Spawn worker processes
    processes = []
    for worker_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = Process(
            target=worker_main,
            args=(worker_id, run_id, chunk, api_key, concurrency),
        )
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()


# ── Combine ──────────────────────────────────────────────────────────────────


def combine_all(valid_entries: list[tuple[int, dict, Path]]):
    """Merge run 0 (original) + runs 1-9 into per-question files with 10 answers."""
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs: dict[int, Path] = {0: ORIGINAL_DIR}
    for run_id in range(1, NUM_NEW_RUNS + 1):
        d = MULTI_DIR / f"run_{run_id}"
        if d.exists():
            run_dirs[run_id] = d

    valid_idxs = {idx for idx, _, _ in valid_entries}
    combined_count = 0
    skipped = 0

    for idx in sorted(valid_idxs):
        out_path = COMBINED_DIR / f"{idx}.json"

        answers = []
        for run_id in sorted(run_dirs.keys()):
            src = run_dirs[run_id] / f"{idx}.json"
            if src.exists():
                with open(src) as f:
                    data = json.load(f)
                if "_meta" in data:
                    data["_meta"]["run_id"] = run_id
                answers.append(data)

        if not answers:
            skipped += 1
            continue

        combined = {
            "idx": idx,
            "num_answers": len(answers),
            "answers": answers,
        }
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        combined_count += 1

    print(f"\nCombined {combined_count} questions into {COMBINED_DIR.name}/")
    counts: dict[int, int] = {}
    for p in COMBINED_DIR.glob("*.json"):
        with open(p) as f:
            n = json.load(f)["num_answers"]
        counts[n] = counts.get(n, 0) + 1
    for n in sorted(counts):
        print(f"  {counts[n]} questions with {n} answer(s)")
    if skipped:
        print(f"  {skipped} questions had no answers at all")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combine", action="store_true", help="Only combine, skip API calls")
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated run IDs to execute, e.g. '1,2,3' (default: all 1-9)")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT,
                        help=f"Max concurrent API requests per worker (default: {MAX_CONCURRENT})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of worker processes per run (default: {DEFAULT_WORKERS})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of questions per run (0=all, useful for benchmarking)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="DeepSeek API key (overrides env var and default)")
    args = parser.parse_args()

    valid_entries = load_valid_entries()
    print(f"Valid questions: {len(valid_entries)}")

    if not args.combine:
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "sk-04915772c096453897b2264c0f057039")
        if not api_key:
            print("Error: set DEEPSEEK_API_KEY environment variable")
            return

        if args.runs:
            run_ids = [int(x.strip()) for x in args.runs.split(",")]
        else:
            run_ids = list(range(1, NUM_NEW_RUNS + 1))

        for run_id in run_ids:
            run_with_workers(
                run_id, valid_entries, api_key,
                num_workers=args.workers,
                concurrency=args.concurrency,
                limit=args.limit,
            )

    combine_all(valid_entries)


if __name__ == "__main__":
    main()
