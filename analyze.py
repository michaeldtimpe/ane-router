"""Summarize results/raw/*.json into comparison matrices.

Usage:
    PYTHONPATH=. .venv-312/bin/python analyze.py [shape_filter] [--md]

--md emits Markdown tables suitable for pasting into the README/summary.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results" / "raw"


def load_all() -> list[dict]:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "null_experiments.json":
            continue
        if path.name == ".gitkeep":
            continue
        try:
            with path.open() as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"  skip {path.name}: {e}")
    return rows


def label_for(r: dict) -> str:
    bk = r["backend"]
    unit = r.get("compute_unit") or ""
    if bk == "coreml":
        return f"coreml/{unit}"
    return bk


_BACKEND_ORDER = {
    "torch_cpu_fp32": 0,
    "torch_cpu_fp16": 1,
    "mlx": 2,
    "coreml": 3,
    "hybrid_all_mlx": 10,
    "hybrid_router_ane": 11,
    "hybrid_expert_ane": 12,
    "hybrid_all_coreml": 13,
}


def sort_key(r: dict) -> tuple:
    return (
        _BACKEND_ORDER.get(r["backend"], 99),
        r.get("compute_unit") or "",
    )


def winner_per_cell(cells: list[dict]) -> dict | None:
    """Pick the lowest-median-latency cell. None if no rows."""
    if not cells:
        return None
    return min(cells, key=lambda r: r["timing"]["median_ms"])


def render_text(by_cell, null_rows):
    for (graph, shape, batch), cells in sorted(by_cell.items()):
        print(f"\n===== [{graph}] {shape}  B={batch} =====")
        nb = null_rows.get(batch)
        if nb:
            print(
                f"  null calibration: boundary={nb['boundary_us']:.2f}us  "
                f"dispatch={nb['dispatch_us']:.2f}us  "
                f"(symmetric hidden=4096 IO; over-counts for asymmetric models)"
            )
        cells.sort(key=sort_key)
        win = winner_per_cell(cells)
        cols = ["backend/unit", "median_us", "p95_us", "first_ms", "actual"]
        print(f"  {cols[0]:<28} {cols[1]:>10} {cols[2]:>10} {cols[3]:>10}  {cols[4]}")
        for r in cells:
            t = r["timing"]
            arrow = " <- min" if r is win else ""
            print(
                f"  {label_for(r):<28} "
                f"{t['median_ms']*1000:>10.2f} "
                f"{t['p95_ms']*1000:>10.2f} "
                f"{t['first_call_ms']:>10.3f}  "
                f"{r.get('actual_execution','?')}{arrow}"
            )


def render_md(by_cell, null_rows):
    print("# Benchmark matrix")
    for (graph, shape, batch), cells in sorted(by_cell.items()):
        print(f"\n## [{graph}] {shape}  B={batch}")
        nb = null_rows.get(batch)
        if nb:
            print(f"_null calibration: boundary={nb['boundary_us']:.2f}µs, "
                  f"dispatch={nb['dispatch_us']:.2f}µs (symmetric h=4096)_\n")
        cells.sort(key=sort_key)
        win = winner_per_cell(cells)
        print("| backend/unit | median µs | p95 µs | first ms | actual |")
        print("|---|---:|---:|---:|---|")
        for r in cells:
            t = r["timing"]
            mark = " ⭐" if r is win else ""
            print(
                f"| {label_for(r)}{mark} "
                f"| {t['median_ms']*1000:.2f} "
                f"| {t['p95_ms']*1000:.2f} "
                f"| {t['first_call_ms']:.3f} "
                f"| {r.get('actual_execution','?')} |"
            )


def main():
    shape_filter = None
    fmt = "text"
    for arg in sys.argv[1:]:
        if arg == "--md":
            fmt = "md"
        else:
            shape_filter = arg

    rows = load_all()
    by_cell: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        if "shape" not in r or "batch" not in r or "timing" not in r:
            continue
        if shape_filter and r["shape"] != shape_filter:
            continue
        graph = r.get("graph", "?")
        by_cell[(graph, r["shape"], r["batch"])].append(r)

    null_path = RESULTS_DIR / "null_experiments.json"
    null_rows: dict[int, dict] = {}
    if null_path.exists():
        with null_path.open() as f:
            for r in json.load(f).get("rows", []):
                null_rows[r["batch"]] = r

    if fmt == "md":
        render_md(by_cell, null_rows)
    else:
        render_text(by_cell, null_rows)


if __name__ == "__main__":
    main()
