"""Energy measurement for the headline router cells.

Workflow:
  1. Run a 5 s warmup with the workload.
  2. Sample 30 s of steady-state with idle-baseline-subtracted PowerSampler.
  3. Compute mJ/token = (steady_mW - idle_mW) * window_s / total_tokens.

Run:
    PYTHONPATH=. .venv-312/bin/python bench/energy_run.py [shape] [batch] [unit]

If args omitted, runs three headline cells:
  router_stress_dense / B=128 / CPU_AND_NE   (ANE actively engaged)
  router_stress_dense / B=128 / CPU_ONLY     (CPU baseline)
  mixtral-8x7b        / B=1   / CPU_AND_NE   (small shape; almost certainly CPU fallback)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from bench.measure import (
    MODELS_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    PowerSampler,
    idle_baseline_mw,
)
from bench.shapes import router_by_name


HEADLINE_CELLS = [
    ("router_stress_dense", 128, "CPU_AND_NE"),
    ("router_stress_dense", 128, "CPU_ONLY"),
    ("mixtral-8x7b", 1, "CPU_AND_NE"),
]


def run_one(shape_name: str, batch: int, unit_name: str,
            warmup_s: float = 5.0, steady_s: float = 30.0,
            sample_ms: int = 200) -> dict:
    """Run the workload for steady_s and return power+latency stats."""
    import coremltools as ct
    shape = router_by_name(shape_name)

    pkg = MODELS_DIR / f"router_{shape.name}_b{batch}.mlpackage"
    if not pkg.exists():
        # Build it
        from bench.router_coreml import _convert
        log = RESULTS_DIR / f"energy_{shape.name}_b{batch}.coreml_warnings.log"
        pkg = _convert(shape, batch, log)

    UNITS = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    model = ct.models.MLModel(str(pkg), compute_units=UNITS[unit_name])
    x = np.random.randn(batch, shape.hidden, 1, 1).astype(np.float16)

    # Warmup
    deadline = time.monotonic() + warmup_s
    n_warm = 0
    while time.monotonic() < deadline:
        model.predict({"x": x})
        n_warm += 1

    # Steady-state with PowerSampler.
    ps = PowerSampler(interval_ms=sample_ms)
    ps.start()
    ps.mark_steady_start()
    deadline = time.monotonic() + steady_s
    n_iter = 0
    t0 = time.perf_counter()
    while time.monotonic() < deadline:
        model.predict({"x": x})
        n_iter += 1
    elapsed = time.perf_counter() - t0
    ps.mark_steady_stop()
    ps.stop()
    steady = ps.steady_state_mw()

    return {
        "shape": shape_name,
        "batch": batch,
        "unit": unit_name,
        "warmup_iters": n_warm,
        "steady_iters": n_iter,
        "steady_s": elapsed,
        "iter_per_s": n_iter / elapsed,
        "us_per_iter": elapsed * 1e6 / n_iter,
        "us_per_token": elapsed * 1e6 / (n_iter * batch),
        "steady_cpu_mw": steady["cpu_mw"],
        "steady_gpu_mw": steady["gpu_mw"],
        "steady_ane_mw": steady["ane_mw"],
        "n_samples": steady["n_samples"],
    }


def main():
    if len(sys.argv) >= 4:
        cells = [(sys.argv[1], int(sys.argv[2]), sys.argv[3])]
    else:
        cells = HEADLINE_CELLS

    print("=== Idle baseline ===")
    idle = idle_baseline_mw(seconds=10, interval_ms=200)
    print(f"  cpu={idle['cpu_mw']:.1f} mW  gpu={idle['gpu_mw']:.1f} mW  "
          f"ane={idle['ane_mw']:.1f} mW  n={idle['n_samples']}")

    rows = []
    for shape, batch, unit in cells:
        print(f"\n=== {shape} B={batch} unit={unit} ===")
        try:
            r = run_one(shape, batch, unit)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue
        delta_cpu = max(0.0, r["steady_cpu_mw"] - idle["cpu_mw"])
        delta_gpu = max(0.0, r["steady_gpu_mw"] - idle["gpu_mw"])
        delta_ane = max(0.0, r["steady_ane_mw"] - idle["ane_mw"])
        delta_total = delta_cpu + delta_gpu + delta_ane
        # mJ/token = mW * (steady_s) / total_tokens / 1000 * 1000
        # = (mW * steady_s) / total_tokens
        # but with mW * s = mJ
        total_tokens = r["steady_iters"] * r["batch"]
        mj_per_token = (delta_total / 1000.0 * r["steady_s"]) / total_tokens * 1000.0
        # Above: (mW/1000=W) * s = J, *1000 = mJ. Then /total_tokens.
        # Simpler: mJ/token = (delta_mW * steady_s) / (total_tokens)
        mj_per_token = (delta_total * r["steady_s"]) / total_tokens

        r["idle_cpu_mw"] = idle["cpu_mw"]
        r["idle_gpu_mw"] = idle["gpu_mw"]
        r["idle_ane_mw"] = idle["ane_mw"]
        r["delta_cpu_mw"] = delta_cpu
        r["delta_gpu_mw"] = delta_gpu
        r["delta_ane_mw"] = delta_ane
        r["delta_total_mw"] = delta_total
        r["mj_per_token"] = mj_per_token
        rows.append(r)

        print(f"  iters: {r['steady_iters']}  ({r['us_per_iter']:.1f} us/iter, "
              f"{r['us_per_token']:.2f} us/token)")
        print(f"  steady: cpu={r['steady_cpu_mw']:.0f} gpu={r['steady_gpu_mw']:.0f} "
              f"ane={r['steady_ane_mw']:.0f} mW  (n={r['n_samples']} samples)")
        print(f"  delta:  cpu={delta_cpu:.0f} gpu={delta_gpu:.0f} ane={delta_ane:.0f} "
              f"= total {delta_total:.0f} mW")
        print(f"  mJ/token: {mj_per_token:.4f}")
        # Per-device sanity check: which rail dominates?
        labels = []
        for label, val in [("ANE", delta_ane), ("GPU", delta_gpu), ("CPU", delta_cpu)]:
            if val > 0.5 * delta_total:
                labels.append(label)
        actual = "MIXED" if len(labels) > 1 else (labels[0] if labels else "?")
        print(f"  inferred actual_execution: {actual}")

    # Persist
    import json
    out = REPO_ROOT / "results" / "raw" / "energy_headline.json"
    with out.open("w") as f:
        json.dump({"idle": idle, "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
