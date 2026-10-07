"""
download_all.py

Downloads ALL 40,336 PhysioNet 2019 training files over plain HTTPS:
  set A: p000001 - p020336   (20,336 patients)
  set B: p100001 - p120000   (20,000 patients)

Uses a few parallel workers so it finishes in a reasonable time. Safe to stop
(Ctrl+C) and rerun: files already downloaded are skipped, and each file is
written to a temporary name first so a stop never leaves a half-written .psv.

Usage (from the project root):
    python src/data_pipeline/download_all.py
"""

import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training"
OUT = Path("data/raw")

# (folder name, first patient number, last patient number)
RANGES = [("training_setA", 1, 20643), ("training_setB", 100001, 120000)]
WORKERS = 6  # lower this (e.g. 3) if your laptop struggles


def fetch(job):
    """Download one patient file. Returns 'ok', 'skipped', or the file name if it failed."""
    folder, n = job
    name = f"p{n:06d}.psv"
    dest = OUT / folder / name
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    for attempt in range(3):  # retry a couple of times on network hiccups
        try:
            with urllib.request.urlopen(f"{BASE}/{folder}/{name}", timeout=30) as r:
                data = r.read()
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)  # rename only after the full file is written
            return "ok"
        except Exception:
            time.sleep(1 + attempt)
    return name


def main() -> None:
    jobs = []
    for folder, first, last in RANGES:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
        jobs += [(folder, n) for n in range(first, last + 1)]

    print(f"{len(jobs)} files to check, {WORKERS} workers...")
    ok = skipped = 0
    failed = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, result in enumerate(pool.map(fetch, jobs), 1):
            if result == "ok":
                ok += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed.append(result)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}  (downloaded {ok}, already had {skipped}, failed {len(failed)})")

    print(f"\nDone. Downloaded {ok}, already had {skipped}, failed {len(failed)}.")
    if failed:
        print("Failed files (rerun the script to retry them):")
        print(failed[:20], "..." if len(failed) > 20 else "")


if __name__ == "__main__":
    main()