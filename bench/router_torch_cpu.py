"""PyTorch CPU baseline for the MoE router.

Router = Linear(hidden, experts) + softmax. CPU runs both fp32 and fp16.

Run:
    PYTHONPATH=. .venv-312/bin/python bench/router_torch_cpu.py [shape_name] [batch]

If args omitted, sweeps all ROUTER_SHAPES x BATCH_SIZES.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn

from bench.measure import ResultRow, RESULTS_DIR, benchmark
from bench.shapes import BATCH_SIZES, ROUTER_SHAPES, RouterShape, router_by_name
from bench.null_experiments import load_null


def make_router(hidden: int, experts: int, dtype: torch.dtype) -> nn.Module:
    m = nn.Sequential(
        nn.Linear(hidden, experts, bias=False),
        nn.Softmax(dim=-1),
    ).eval().to(dtype)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def bench_one(shape: RouterShape, batch: int, dtype: torch.dtype, iters: int = 1000) -> ResultRow:
    torch.set_num_threads(8)  # M1 Max P-cores
    model = make_router(shape.hidden, shape.experts, dtype)
    x = torch.randn(batch, shape.hidden, dtype=dtype)
    # Force one forward to warm up Accelerate dispatcher
    with torch.inference_mode():
        model(x)

    @torch.inference_mode()
    def step():
        model(x)

    t = benchmark(step, iters=iters, warmup=100)
    null = load_null(batch) or {}
    boundary_us = null.get("boundary_us", 0.0)
    # CPU path has no CoreML; dispatch_us is irrelevant.
    kernel_us = max(0.0, t.median_ms * 1000.0)

    precision = "fp32" if dtype == torch.float32 else "fp16"
    return ResultRow(
        shape=shape.name,
        graph="router_only",
        backend=f"torch_cpu_{precision}",
        compute_unit="CPU_ONLY",
        actual_execution="CPU",
        batch=batch,
        precision=precision,
        timing=t,
        boundary_us=boundary_us,
        dispatch_us=0.0,
        kernel_us=kernel_us,
    )


def main():
    if len(sys.argv) >= 3:
        shapes = [router_by_name(sys.argv[1])]
        batches = [int(sys.argv[2])]
    else:
        shapes = ROUTER_SHAPES
        batches = BATCH_SIZES

    for shape in shapes:
        for b in batches:
            for dtype in (torch.float32, torch.float16):
                row = bench_one(shape, b, dtype)
                path = row.write()
                tag = f"{row.backend} {shape.name} B={b}"
                print(f"  {tag:<48} median={row.timing.median_ms*1000:>9.2f}us  "
                      f"p95={row.timing.p95_ms*1000:>9.2f}us")
                print(f"    -> {path.name}")


if __name__ == "__main__":
    main()
