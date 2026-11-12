"""Preflight of a clean rerun: refuse an ``outputs/`` tree that holds anything but ``logs/``, or list what it holds.

    python -m scripts.preflight            # --fresh: exit non-zero unless outputs/ holds no file outside logs/
    python -m scripts.preflight --inventory  # print, per cache kind, what exists and whether a stage would reuse it

Standard library only, so it runs before anything else is importable. The
first link of every batch chain: a rerun must never reuse or mix an old
checkpoint, shard, split or table. Walk and cross-fit fingerprints cover the
configuration and the operator, not the source code, so a stale cache can
only be excluded by starting from an empty tree. A directory with no file
below it (what ``scripts.warm --list`` or a test run can leave behind) cannot
hold a cache and does not count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "results" / "outputs"
LOGS = OUTPUTS / "logs"

# (label, glob under outputs/, how the stage treats an existing file)
CACHE_KINDS = (
    ("quantum walk checkpoints (stage 04 reduction check)", "notebook_pipeline/*/checkpoints/*/completed.npz",
     "reused when the configuration and operator fingerprint match"),
    ("partial walk checkpoints", "notebook_pipeline/*/checkpoints/*/latest.npz", "resumed when the fingerprint matches"),
    ("cross-fit fold scores (stage 03)", "notebook_pipeline/*/fold_*.npy", "reused when the cache key matches"),
    ("hardware noise-grid shards (stage 06)", "notebook_pipeline/*/grid_shards/grid_*.json", "reused when the shard identity matches; a stale identity raises"),
    ("frozen splits and bootstrap matrices (stage 01)", "*frozen_split.json", "rewritten by stage 01"),
    ("bootstrap index matrices (stage 01)", "*bootstrap_indices.npy", "rewritten by stage 01"),
    ("masked frames", "notebook_pipeline/masked/*/*.parquet", "rewritten by stage 03"),
    ("operator digests", "notebook_pipeline/*/operator_digests.json", "rewritten by stages 03 and 04"),
    ("result tables", "results/*/*.csv", "rewritten by the stage that owns them"),
    ("manuscript tables, figures and numbers", "manuscript/**/*", "rewritten by the assembler scripts"),
    ("stage manifests", "notebook_pipeline/*/manifest.json", "rewritten by every stage"),
    ("executed notebooks", "executed_notebooks/*.ipynb", "rewritten by run_all"),
)


def holds_a_file(entry: Path) -> bool:
    """A file, or a directory with a file somewhere below it: an empty directory tree cannot hold a stale cache."""
    return entry.is_file() or any(p.is_file() for p in entry.rglob("*"))


def offending_entries() -> list[Path]:
    """Everything under ``outputs/`` that holds a file, except the ``logs/`` directory and its contents."""
    if not OUTPUTS.exists():
        return []
    return sorted(p for p in OUTPUTS.iterdir() if p != LOGS and holds_a_file(p))


def fresh() -> int:
    entries = offending_entries()
    if not entries:
        print(f"preflight: {OUTPUTS.relative_to(ROOT)}/ holds nothing but logs/ (and empty directories); clean start")
        return 0
    print(f"preflight: {OUTPUTS.relative_to(ROOT)}/ is not clean; delete it (keep data/, .venv/, notebooks/, scripts/*.sbatch) "
          "and recreate outputs/logs before submitting:", file=sys.stderr)
    for entry in entries:
        count = sum(1 for _ in entry.rglob("*")) if entry.is_dir() else 0
        kind = f"directory with {count} entries" if entry.is_dir() else "file"
        print(f"  {entry.relative_to(ROOT)}  ({kind})", file=sys.stderr)
    return 1


def inventory() -> int:
    print(f"preflight inventory of {OUTPUTS.relative_to(ROOT)}/")
    if not OUTPUTS.exists():
        print("  (outputs/ does not exist)")
        return 0
    for label, pattern, policy in CACHE_KINDS:
        matches = sorted(p for p in OUTPUTS.glob(pattern) if p.is_file())
        print(f"- {label}: {len(matches)} file(s); {policy}")
        for p in matches[:12]:
            print(f"    {p.relative_to(ROOT)}")
        if len(matches) > 12:
            print(f"    ... {len(matches) - 12} more")
    data = ROOT / "data"
    processed = sorted(data.glob("*_processed.parquet")) if data.exists() else []
    print(f"- processed frames under data/ (survive a deletion of outputs/): {len(processed)} file(s); stage 01 rewrites them")
    for p in processed:
        print(f"    {p.relative_to(ROOT)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fresh", action="store_true", help="exit non-zero unless outputs/ holds nothing but logs/ (default)")
    group.add_argument("--inventory", action="store_true", help="list what exists per cache kind; never fails")
    args = parser.parse_args(argv)
    return inventory() if args.inventory else fresh()


if __name__ == "__main__":
    raise SystemExit(main())
