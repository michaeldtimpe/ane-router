"""CoreML expert MLP, with ComputeUnit sweep.

Expert in ANE-friendly bc1s layout: (B, hidden, 1, 1) -> (B, hidden, 1, 1).
Each Linear is a Conv2d 1x1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from bench.measure import (
    MODELS_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    ResultRow,
    benchmark,
    capture_coreml_warnings,
)
from bench.null_experiments import load_null
from bench.shapes import BATCH_SIZES, EXPERT_SHAPES, ExpertShape, expert_by_name

UNITS = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "ALL": ct.ComputeUnit.ALL,
}


class ExpertANE(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.up = nn.Conv2d(hidden, ffn, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(hidden, ffn, kernel_size=1, bias=False)
        self.down = nn.Conv2d(ffn, hidden, kernel_size=1, bias=False)

    def forward(self, x):
        u = self.up(x)
        g = torch.nn.functional.silu(self.gate(x))
        return self.down(u * g)


def _convert(shape: ExpertShape, batch: int, log_path: Path) -> Path:
    cache = MODELS_DIR / f"expert_{shape.name}_b{batch}.mlpackage"
    if cache.exists():
        return cache

    model = ExpertANE(shape.hidden, shape.ffn).eval().half()
    for p in model.parameters():
        p.requires_grad_(False)
    ex = torch.randn(batch, shape.hidden, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(model, ex)
    with capture_coreml_warnings(log_path):
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=ex.shape, name="x", dtype=np.float16)],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
        )
    ml.save(str(cache))
    return cache


def bench_one(shape: ExpertShape, batch: int, unit_name: str, iters: int = 200) -> ResultRow:
    log_path = RESULTS_DIR / f"expert_{shape.name}_b{batch}.coreml_warnings.log"
    pkg = _convert(shape, batch, log_path)
    unit = UNITS[unit_name]
    model = ct.models.MLModel(str(pkg), compute_units=unit)
    x = np.random.randn(batch, shape.hidden, 1, 1).astype(np.float16)
    # ANE may reject a model at runtime even after the compiler accepts it;
    # capture as a result rather than crash the sweep.
    try:
        model.predict({"x": x})
    except RuntimeError as e:
        msg = str(e)
        if "ANE" in msg or "appleneuralengine" in msg or "appleneural" in msg.lower():
            from bench.measure import TimingResult
            stub = TimingResult(median_ms=0.0, p95_ms=0.0, p99_ms=0.0, stddev_ms=0.0,
                                first_call_ms=0.0, n_iters=0, n_warmup=0)
            return ResultRow(
                shape=shape.name,
                graph="expert",
                backend="coreml",
                compute_unit=unit_name,
                actual_execution="ANE_INFERENCE_ERROR",
                batch=batch,
                precision="fp16",
                timing=stub,
                boundary_us=0.0, dispatch_us=0.0, kernel_us=0.0,
                coreml_warnings_log=str(log_path.relative_to(REPO_ROOT)),
                notes=f"ANE rejected this model at inference: {msg[:200]}",
            )
        raise

    def step():
        model.predict({"x": x})

    t = benchmark(step, iters=iters, warmup=20)
    null = load_null(batch) or {}
    boundary_us = null.get("boundary_us", 0.0)
    dispatch_us = null.get("dispatch_us", 0.0)
    total_us = t.median_ms * 1000.0
    return ResultRow(
        shape=shape.name,
        graph="expert",
        backend="coreml",
        compute_unit=unit_name,
        actual_execution="UNKNOWN",
        batch=batch,
        precision="fp16",
        timing=t,
        boundary_us=boundary_us,
        dispatch_us=dispatch_us,
        kernel_us=max(0.0, total_us - boundary_us - dispatch_us),
        coreml_warnings_log=str(log_path.relative_to(REPO_ROOT)),
        notes="actual_execution must be populated from Instruments Core ML trace",
    )


def main():
    if len(sys.argv) >= 4:
        shapes = [expert_by_name(sys.argv[1])]
        batches = [int(sys.argv[2])]
        units = [sys.argv[3]]
    elif len(sys.argv) >= 3:
        shapes = [expert_by_name(sys.argv[1])]
        batches = [int(sys.argv[2])]
        units = list(UNITS.keys())
    else:
        shapes = EXPERT_SHAPES
        batches = BATCH_SIZES
        units = list(UNITS.keys())

    for shape in shapes:
        for b in batches:
            for u in units:
                row = bench_one(shape, b, u)
                row.write()
                tag = f"coreml/{u} expert/{shape.name} B={b}"
                print(f"  {tag:<48} median={row.timing.median_ms:>9.3f}ms  "
                      f"p95={row.timing.p95_ms:>9.3f}ms  "
                      f"kernel={row.kernel_us/1000:>8.3f}ms  first={row.timing.first_call_ms:.3f}ms")


if __name__ == "__main__":
    main()
