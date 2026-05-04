"""Null calibration experiments.

These produce the floor numbers for any hybrid configuration:
  boundary_us — MLX -> numpy -> MLX, no CoreML
  dispatch_us — MLX -> CoreML identity -> MLX, minus boundary

Every hybrid result row gets these two numbers attached so a slow hybrid
can be attributed (boundary vs dispatch vs kernel vs ε), instead of being
mysterious.

Run:
    .venv-312/bin/python bench/null_experiments.py
Writes results/raw/null_experiments.json.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from bench.measure import (
    REPO_ROOT,
    RESULTS_DIR,
    HostInfo,
    measure_coreml_empty,
    measure_null_roundtrip,
)
from bench.shapes import BATCH_SIZES


def main() -> None:
    print("=== Null calibration ===")
    print(f"batches: {BATCH_SIZES}")

    out: dict = {
        "host": asdict(HostInfo.collect()),
        "hidden_used_for_calibration": 4096,
        "rows": [],
    }

    # Reference hidden dim for null measurement (matches Mixtral router input).
    HIDDEN = 4096
    ITERS = 1000

    for b in BATCH_SIZES:
        print(f"\n--- batch={b} ---")

        print("  null_roundtrip (MLX -> numpy -> MLX) ...", end=" ", flush=True)
        nr = measure_null_roundtrip(batch=b, hidden=HIDDEN, iters=ITERS)
        boundary_us = nr.median_ms * 1000.0
        print(f"median={nr.median_ms:.4f} ms (p95={nr.p95_ms:.4f} ms)")

        print("  coreml_empty   (MLX -> CoreML identity -> MLX) ...", end=" ", flush=True)
        ce, label = measure_coreml_empty(batch=b, hidden=HIDDEN, iters=ITERS)
        empty_us = ce.median_ms * 1000.0
        dispatch_us = max(0.0, empty_us - boundary_us)
        print(f"median={ce.median_ms:.4f} ms (p95={ce.p95_ms:.4f} ms)")
        print(f"  -> dispatch_us = {dispatch_us:.2f} (boundary={boundary_us:.2f})")

        out["rows"].append({
            "batch": b,
            "hidden": HIDDEN,
            "null_roundtrip": asdict(nr),
            "coreml_empty": asdict(ce),
            "boundary_us": boundary_us,
            "dispatch_us": dispatch_us,
            "actual_execution_coreml_empty": label,
        })

    path = RESULTS_DIR / "null_experiments.json"
    with path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


def load_null(batch: int) -> dict | None:
    """Helper for other benchmarks to look up calibration values."""
    path = RESULTS_DIR / "null_experiments.json"
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    for row in data.get("rows", []):
        if row.get("batch") == batch:
            return row
    return None


if __name__ == "__main__":
    main()
