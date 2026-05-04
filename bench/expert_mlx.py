"""MLX expert MLP baseline (Metal/GPU)."""
from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

from bench.measure import ResultRow, benchmark
from bench.null_experiments import load_null
from bench.shapes import BATCH_SIZES, EXPERT_SHAPES, ExpertShape, expert_by_name


def bench_one(shape: ExpertShape, batch: int, iters: int = 200) -> ResultRow:
    H, F = shape.hidden, shape.ffn
    W_up = mx.array(np.random.randn(H, F).astype(np.float16) * 0.02)
    W_gate = mx.array(np.random.randn(H, F).astype(np.float16) * 0.02)
    W_down = mx.array(np.random.randn(F, H).astype(np.float16) * 0.02)
    x = mx.array(np.random.randn(batch, H).astype(np.float16))
    mx.eval(W_up, W_gate, W_down, x)

    def fwd():
        u = x @ W_up
        g = mx.sigmoid(x @ W_gate) * (x @ W_gate)  # silu
        return (u * g) @ W_down

    # Warmup the GPU.
    y = fwd(); mx.eval(y)

    def step():
        y = fwd()
        mx.eval(y)

    t = benchmark(step, iters=iters, warmup=20)
    null = load_null(batch) or {}
    return ResultRow(
        shape=shape.name,
        graph="expert",
        backend="mlx",
        compute_unit="CPU_AND_GPU",
        actual_execution="GPU",
        batch=batch,
        precision="fp16",
        timing=t,
        boundary_us=null.get("boundary_us", 0.0),
        dispatch_us=0.0,
        kernel_us=t.median_ms * 1000.0,
    )


def main():
    if len(sys.argv) >= 3:
        shapes = [expert_by_name(sys.argv[1])]
        batches = [int(sys.argv[2])]
    else:
        shapes = EXPERT_SHAPES
        batches = BATCH_SIZES

    for shape in shapes:
        for b in batches:
            row = bench_one(shape, b)
            row.write()
            tag = f"mlx expert/{shape.name} B={b}"
            print(f"  {tag:<40} median={row.timing.median_ms:>9.3f}ms  "
                  f"p95={row.timing.p95_ms:>9.3f}ms  first={row.timing.first_call_ms:.3f}ms")


if __name__ == "__main__":
    main()
