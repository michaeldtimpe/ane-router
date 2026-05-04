"""MLX (Metal/GPU) baseline for the MoE router.

Router = Linear(hidden, experts) + softmax, fp16, on the M1 GPU.

Run:
    PYTHONPATH=. .venv-312/bin/python bench/router_mlx.py [shape_name] [batch]
"""
from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

from bench.measure import ResultRow, benchmark
from bench.null_experiments import load_null
from bench.shapes import BATCH_SIZES, ROUTER_SHAPES, RouterShape, router_by_name


def bench_one(shape: RouterShape, batch: int, iters: int = 1000) -> ResultRow:
    # Direct fp16 weights and matmul. Avoids any framework upcasting.
    W = mx.array(
        np.random.randn(shape.hidden, shape.experts).astype(np.float16) * 0.02
    )
    x = mx.array(np.random.randn(batch, shape.hidden).astype(np.float16))
    mx.eval(W, x)

    # Warm GPU caches.
    y = mx.softmax(x @ W, axis=-1); mx.eval(y)

    def step():
        y = mx.softmax(x @ W, axis=-1)
        mx.eval(y)

    t = benchmark(step, iters=iters, warmup=100)
    null = load_null(batch) or {}
    boundary_us = null.get("boundary_us", 0.0)
    return ResultRow(
        shape=shape.name,
        graph="router_only",
        backend="mlx",
        compute_unit="CPU_AND_GPU",  # MLX runs on Metal/GPU on M1
        actual_execution="GPU",
        batch=batch,
        precision="fp16",
        timing=t,
        boundary_us=boundary_us,
        dispatch_us=0.0,
        kernel_us=max(0.0, t.median_ms * 1000.0),
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
            row = bench_one(shape, b)
            path = row.write()
            tag = f"mlx {shape.name} B={b}"
            print(f"  {tag:<40} median={row.timing.median_ms*1000:>9.2f}us  "
                  f"p95={row.timing.p95_ms*1000:>9.2f}us  first={row.timing.first_call_ms:.3f}ms")
            print(f"    -> {path.name}")


if __name__ == "__main__":
    main()
