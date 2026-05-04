"""PyTorch CPU baseline for an MoE expert MLP.

Expert form (Mixtral-style SwiGLU): silu(W1 x) * (W3 x) -> W2 -> y.
Simplified to single MLP for benchmarking compute volume:
  up = W_up @ x   (hidden -> ffn)
  gate = silu(W_gate @ x)
  inter = up * gate
  y = W_down @ inter   (ffn -> hidden)
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from bench.measure import ResultRow, benchmark
from bench.null_experiments import load_null
from bench.shapes import BATCH_SIZES, EXPERT_SHAPES, ExpertShape, expert_by_name


class ExpertMLP(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.w_up = nn.Linear(hidden, ffn, bias=False)
        self.w_gate = nn.Linear(hidden, ffn, bias=False)
        self.w_down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


def bench_one(shape: ExpertShape, batch: int, dtype: torch.dtype, iters: int = 200) -> ResultRow:
    torch.set_num_threads(8)
    model = ExpertMLP(shape.hidden, shape.ffn).eval().to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(batch, shape.hidden, dtype=dtype)
    with torch.inference_mode():
        model(x)

    @torch.inference_mode()
    def step():
        model(x)

    t = benchmark(step, iters=iters, warmup=20)
    null = load_null(batch) or {}
    precision = "fp32" if dtype == torch.float32 else "fp16"
    return ResultRow(
        shape=shape.name,
        graph="expert",
        backend=f"torch_cpu_{precision}",
        compute_unit="CPU_ONLY",
        actual_execution="CPU",
        batch=batch,
        precision=precision,
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
            for dtype in (torch.float32, torch.float16):
                row = bench_one(shape, b, dtype)
                row.write()
                tag = f"{row.backend} expert/{shape.name} B={b}"
                print(f"  {tag:<48} median={row.timing.median_ms:>9.3f}ms  "
                      f"p95={row.timing.p95_ms:>9.3f}ms")


if __name__ == "__main__":
    main()
