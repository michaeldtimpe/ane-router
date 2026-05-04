"""Patch result JSONs with `actual_execution` after Instruments inspection.

Usage:
    .venv-312/bin/python scripts/patch_actual_execution.py \
        --shape mixtral-8x7b --graph router_only --batch 1 \
        --unit CPU_AND_NE --actual ANE_FALLBACK \
        --trace results/instruments/<file>.trace

Walks results/raw/*.json and updates every row matching (shape, graph, batch,
compute_unit). Sets `actual_execution` and `instruments_trace`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "raw"

VALID_ACTUAL = {"ANE", "GPU", "CPU", "MIXED", "ANE_FALLBACK", "UNKNOWN"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True)
    p.add_argument("--graph", default=None,
                   help="Optional: router_only|router_topk|expert|block. "
                        "If omitted, matches all graphs for the shape.")
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--unit", required=True,
                   help="ComputeUnit the row was run with (CPU_AND_NE, ALL, ...).")
    p.add_argument("--actual", required=True, choices=sorted(VALID_ACTUAL))
    p.add_argument("--trace", default=None,
                   help="Optional path to .trace export.")
    args = p.parse_args()

    matched = 0
    for path in RESULTS_DIR.glob("*.json"):
        if path.name == "null_experiments.json":
            continue
        with path.open() as f:
            row = json.load(f)
        if row.get("shape") != args.shape:
            continue
        if args.graph is not None and row.get("graph") != args.graph:
            continue
        if row.get("batch") != args.batch:
            continue
        if row.get("compute_unit") != args.unit:
            continue

        row["actual_execution"] = args.actual
        if args.trace:
            row["instruments_trace"] = args.trace
        with path.open("w") as f:
            json.dump(row, f, indent=2, default=str)
        matched += 1
        print(f"  patched {path.name}")

    print(f"\nmatched {matched} row(s).")


if __name__ == "__main__":
    main()
