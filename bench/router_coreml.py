"""CoreML router benchmark, with ComputeUnit sweep.

Router built as Conv2d(1x1) on bc1s layout (B, C_hidden, 1, 1) so the ANE
compiler can map it to its native form. softmax over channel dim.

Each ComputeUnit setting is a request — the compiler may ignore it. The
recorded `actual_execution` is `UNKNOWN` from this script; populate from
Instruments after the fact.

Run:
    PYTHONPATH=. .venv-312/bin/python bench/router_coreml.py [shape] [batch] [unit]
units: CPU_ONLY | CPU_AND_GPU | CPU_AND_NE | ALL
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
from bench.shapes import BATCH_SIZES, ROUTER_SHAPES, RouterShape, router_by_name

UNITS = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "ALL": ct.ComputeUnit.ALL,
}


class RouterANE(nn.Module):
    """ANE-canonical: input (B, hidden, 1, 1), Conv2d 1x1 maps Linear, softmax over channels."""

    def __init__(self, hidden: int, experts: int):
        super().__init__()
        self.gate = nn.Conv2d(hidden, experts, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, hidden, 1, 1) -> (B, experts, 1, 1)
        logits = self.gate(x)
        return torch.softmax(logits, dim=1)


def _convert(shape: RouterShape, batch: int, log_path: Path) -> Path:
    cache = MODELS_DIR / f"router_{shape.name}_b{batch}.mlpackage"
    if cache.exists():
        return cache

    model = RouterANE(shape.hidden, shape.experts).eval().half()
    for p in model.parameters():
        p.requires_grad_(False)
    ex = torch.randn(batch, shape.hidden, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(model, ex)

    with capture_coreml_warnings(log_path):
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=ex.shape, name="x", dtype=np.float16)],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            compute_units=ct.ComputeUnit.CPU_AND_NE,  # only metadata; runtime arg overrides
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
        )
    ml.save(str(cache))
    return cache


def bench_one(
    shape: RouterShape,
    batch: int,
    unit_name: str,
    iters: int = 1000,
) -> ResultRow:
    log_path = RESULTS_DIR / f"router_{shape.name}_b{batch}.coreml_warnings.log"
    pkg = _convert(shape, batch, log_path)

    unit = UNITS[unit_name]
    model = ct.models.MLModel(str(pkg), compute_units=unit)
    x = np.random.randn(batch, shape.hidden, 1, 1).astype(np.float16)

    # Warm: this also covers the cold first-call dispatch penalty.
    # ANE may reject some shapes at inference time (status=0x15) — capture
    # that as a result rather than crashing the whole sweep.
    try:
        model.predict({"x": x})
    except RuntimeError as e:
        msg = str(e)
        if "ANE" in msg or "appleneuralengine" in msg:
            from bench.measure import TimingResult
            stub = TimingResult(median_ms=0.0, p95_ms=0.0, p99_ms=0.0, stddev_ms=0.0,
                                first_call_ms=0.0, n_iters=0, n_warmup=0)
            return ResultRow(
                shape=shape.name,
                graph="router_only",
                backend="coreml",
                compute_unit=unit_name,
                actual_execution="ANE_INFERENCE_ERROR",
                batch=batch,
                precision="fp16",
                timing=stub,
                boundary_us=0.0,
                dispatch_us=0.0,
                kernel_us=0.0,
                coreml_warnings_log=str(log_path.relative_to(REPO_ROOT)),
                notes=f"ANE rejected this model at inference: {msg[:200]}",
            )
        raise

    def step():
        model.predict({"x": x})

    t = benchmark(step, iters=iters, warmup=100)

    null = load_null(batch) or {}
    boundary_us = null.get("boundary_us", 0.0)
    dispatch_us = null.get("dispatch_us", 0.0)
    total_us = t.median_ms * 1000.0
    kernel_us = max(0.0, total_us - boundary_us - dispatch_us)

    return ResultRow(
        shape=shape.name,
        graph="router_only",
        backend="coreml",
        compute_unit=unit_name,
        actual_execution="UNKNOWN",  # populate from Instruments
        batch=batch,
        precision="fp16",
        timing=t,
        boundary_us=boundary_us,
        dispatch_us=dispatch_us,
        kernel_us=kernel_us,
        coreml_warnings_log=str(log_path.relative_to(REPO_ROOT)),
        notes="actual_execution must be populated from Instruments Core ML trace",
    )


def main():
    if len(sys.argv) >= 4:
        shapes = [router_by_name(sys.argv[1])]
        batches = [int(sys.argv[2])]
        units = [sys.argv[3]]
    elif len(sys.argv) >= 3:
        shapes = [router_by_name(sys.argv[1])]
        batches = [int(sys.argv[2])]
        units = list(UNITS.keys())
    else:
        shapes = ROUTER_SHAPES
        batches = BATCH_SIZES
        units = list(UNITS.keys())

    for shape in shapes:
        for b in batches:
            for u in units:
                row = bench_one(shape, b, u)
                path = row.write()
                tag = f"coreml/{u} {shape.name} B={b}"
                print(f"  {tag:<48} median={row.timing.median_ms*1000:>9.2f}us  "
                      f"p95={row.timing.p95_ms*1000:>9.2f}us  "
                      f"kernel={row.kernel_us:>9.2f}us  first={row.timing.first_call_ms:.3f}ms")
                print(f"    -> {path.name}")


if __name__ == "__main__":
    main()
