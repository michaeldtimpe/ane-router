"""Energy measurement for expert MLP and block-hybrid headline cells.

Mirrors bench/energy_run.py but for the larger ops where ANE actually
engages per the compute plan (MLComputePlan.preferred_compute_device).

Cells run by default:
  expert/mixtral-8x7b   B=32   CPU_ONLY     (CPU baseline at the H2 batch)
  expert/mixtral-8x7b   B=32   CPU_AND_NE   (compute plan says ANE)
  expert/olmoe          B=8    CPU_ONLY
  expert/olmoe          B=8    CPU_AND_NE
"""
from __future__ import annotations

import json
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
from bench.shapes import expert_by_name


HEADLINE_EXPERT_CELLS = [
    ("mixtral-8x7b", 32, "CPU_ONLY"),
    ("mixtral-8x7b", 32, "CPU_AND_NE"),
    ("olmoe", 8, "CPU_ONLY"),
    ("olmoe", 8, "CPU_AND_NE"),
]


def run_one(shape_name: str, batch: int, unit_name: str,
            warmup_s: float = 5.0, steady_s: float = 30.0,
            sample_ms: int = 200) -> dict:
    import coremltools as ct
    shape = expert_by_name(shape_name)

    pkg = MODELS_DIR / f"expert_{shape.name}_b{batch}.mlpackage"
    if not pkg.exists():
        from bench.expert_coreml import _convert
        log = RESULTS_DIR / f"energy_expert_{shape.name}_b{batch}.coreml_warnings.log"
        pkg = _convert(shape, batch, log)

    UNITS = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    model = ct.models.MLModel(str(pkg), compute_units=UNITS[unit_name])
    x = np.random.randn(batch, shape.hidden, 1, 1).astype(np.float16)

    deadline = time.monotonic() + warmup_s
    n_warm = 0
    while time.monotonic() < deadline:
        model.predict({"x": x})
        n_warm += 1

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
        "graph": "expert",
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
        cells = HEADLINE_EXPERT_CELLS

    print("=== Idle baseline (10s) ===")
    idle = idle_baseline_mw(seconds=10, interval_ms=200)
    print(f"  cpu={idle['cpu_mw']:.1f} gpu={idle['gpu_mw']:.1f} "
          f"ane={idle['ane_mw']:.1f} mW  n={idle['n_samples']}")

    rows = []
    for shape, batch, unit in cells:
        print(f"\n=== expert/{shape} B={batch} unit={unit} ===")
        try:
            r = run_one(shape, batch, unit)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue
        d_cpu = max(0.0, r["steady_cpu_mw"] - idle["cpu_mw"])
        d_gpu = max(0.0, r["steady_gpu_mw"] - idle["gpu_mw"])
        d_ane = max(0.0, r["steady_ane_mw"] - idle["ane_mw"])
        d_total = d_cpu + d_gpu + d_ane
        total_tokens = r["steady_iters"] * r["batch"]
        mj_per_token = (d_total * r["steady_s"]) / total_tokens
        r["delta_cpu_mw"] = d_cpu
        r["delta_gpu_mw"] = d_gpu
        r["delta_ane_mw"] = d_ane
        r["delta_total_mw"] = d_total
        r["mj_per_token"] = mj_per_token
        r["idle"] = idle
        rows.append(r)

        print(f"  iters: {r['steady_iters']}  ({r['us_per_iter']:.1f} us/iter, "
              f"{r['us_per_token']:.2f} us/token)")
        print(f"  steady: cpu={r['steady_cpu_mw']:.0f} gpu={r['steady_gpu_mw']:.0f} "
              f"ane={r['steady_ane_mw']:.0f} mW  (n={r['n_samples']})")
        print(f"  delta:  cpu={d_cpu:.0f} gpu={d_gpu:.0f} ane={d_ane:.0f} "
              f"total={d_total:.0f} mW")
        print(f"  mJ/token: {mj_per_token:.4f}")

    out = REPO_ROOT / "results" / "raw" / "energy_expert.json"
    with out.open("w") as f:
        json.dump({"idle": idle, "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
