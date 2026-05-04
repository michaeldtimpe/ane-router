"""THE experiment: full MoE block with all four placement combinations.

For a given block shape (hidden, ffn, experts, top_k=1), we time the simplified
forward pass:

    logits = softmax(x @ W_gate)       # router
    weights, idx = top_k(logits, k=1)  # static k=1 to avoid CPU fallback
    y = expert_idx(x) * weights[:, 0]  # expert forward (top expert only)

Four placements:

    (A) all-MLX            router on MLX, expert on MLX
    (B) router-CoreML/expert-MLX
    (C) router-MLX/expert-CoreML
    (D) all-CoreML         single compiled CoreML graph for the whole block

(A) has zero MLX<->CoreML boundary crossings.
(B) and (C) pay one round-trip per call (the side that's on CoreML).
(D) compiles router+expert as one CoreML graph: zero per-call crossings.

For (B) and (C), only ONE expert is materialized as the candidate (real MoE
gathers from many; here we strip the gather to isolate placement cost).
"""
from __future__ import annotations

import sys
from pathlib import Path

import coremltools as ct
import mlx.core as mx
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
from bench.shapes import BLOCK_SHAPES, BlockShape, block_by_name


# ---------------------------------------------------------------------------
# CoreML pieces
# ---------------------------------------------------------------------------


class _RouterANE(nn.Module):
    def __init__(self, hidden: int, experts: int):
        super().__init__()
        self.gate = nn.Conv2d(hidden, experts, kernel_size=1, bias=False)

    def forward(self, x):
        return torch.softmax(self.gate(x), dim=1)


class _ExpertANE(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.up = nn.Conv2d(hidden, ffn, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(hidden, ffn, kernel_size=1, bias=False)
        self.down = nn.Conv2d(ffn, hidden, kernel_size=1, bias=False)

    def forward(self, x):
        u = self.up(x)
        g = torch.nn.functional.silu(self.gate(x))
        return self.down(u * g)


class _BlockANE(nn.Module):
    """Single-graph block: router + one fixed expert. Realistic enough for
    benchmarking the placement question; gather/scatter logic skipped.
    """

    def __init__(self, hidden: int, ffn: int, experts: int):
        super().__init__()
        self.router = _RouterANE(hidden, experts)
        self.expert = _ExpertANE(hidden, ffn)

    def forward(self, x):
        # Use the first router weight as the scaling for the expert output.
        w = self.router(x)[:, :1]   # (B, 1, 1, 1)
        return self.expert(x) * w


def _convert(module: nn.Module, name: str, batch: int, hidden: int, log_path: Path) -> Path:
    cache = MODELS_DIR / f"{name}_b{batch}.mlpackage"
    if cache.exists():
        return cache
    module = module.eval().half()
    for p in module.parameters():
        p.requires_grad_(False)
    ex = torch.randn(batch, hidden, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(module, ex)
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


# ---------------------------------------------------------------------------
# Placements
# ---------------------------------------------------------------------------


def _make_mlx_weights(shape: BlockShape):
    H, F, E = shape.hidden, shape.ffn, shape.experts
    W_gate = mx.array(np.random.randn(H, E).astype(np.float16) * 0.02)
    W_up = mx.array(np.random.randn(H, F).astype(np.float16) * 0.02)
    W_g = mx.array(np.random.randn(H, F).astype(np.float16) * 0.02)
    W_down = mx.array(np.random.randn(F, H).astype(np.float16) * 0.02)
    mx.eval(W_gate, W_up, W_g, W_down)
    return W_gate, W_up, W_g, W_down


def placement_A_all_mlx(shape: BlockShape, batch: int, iters: int = 200) -> ResultRow:
    """All-MLX, the GPU-only baseline. No boundary crossings."""
    W_gate, W_up, W_g, W_down = _make_mlx_weights(shape)
    x = mx.array(np.random.randn(batch, shape.hidden).astype(np.float16))
    mx.eval(x)

    def fwd():
        logits = mx.softmax(x @ W_gate, axis=-1)
        w = logits[:, :1]   # (B, 1)
        u = x @ W_up
        g = mx.sigmoid(x @ W_g) * (x @ W_g)  # silu
        return ((u * g) @ W_down) * w

    y = fwd(); mx.eval(y)

    def step():
        y = fwd()
        mx.eval(y)

    t = benchmark(step, iters=iters, warmup=20)
    null = load_null(batch) or {}
    return ResultRow(
        shape=shape.name,
        graph="block",
        backend="hybrid_all_mlx",
        compute_unit="CPU_AND_GPU",
        actual_execution="GPU",
        batch=batch,
        precision="fp16",
        timing=t,
        boundary_us=null.get("boundary_us", 0.0),
        dispatch_us=0.0,
        kernel_us=t.median_ms * 1000.0,
        notes="all-MLX baseline (no boundary crossings)",
    )


def placement_B_router_coreml(shape: BlockShape, batch: int, iters: int = 200) -> ResultRow:
    """Router on CoreML, expert on MLX. Pays one boundary round-trip per call."""
    log = RESULTS_DIR / f"block_router_{shape.name}_b{batch}.coreml_warnings.log"
    pkg = _convert(_RouterANE(shape.hidden, shape.experts),
                   f"block_router_{shape.name}", batch, shape.hidden, log)
    router = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)

    _, W_up, W_g, W_down = _make_mlx_weights(shape)
    x = mx.array(np.random.randn(batch, shape.hidden).astype(np.float16))
    mx.eval(x)

    def fwd():
        # Bridge x -> CoreML -> bridge back
        x_np = np.asarray(x, dtype=np.float16).reshape(batch, shape.hidden, 1, 1)
        logits = router.predict({"x": x_np})["y"]  # (B, E, 1, 1)
        w = mx.array(logits[:, :1].reshape(batch, 1))
        u = x @ W_up
        g = mx.sigmoid(x @ W_g) * (x @ W_g)
        out = ((u * g) @ W_down) * w
        mx.eval(out)
        return out

    fwd()  # warm

    def step():
        fwd()

    t = benchmark(step, iters=iters, warmup=20)
    null = load_null(batch) or {}
    return ResultRow(
        shape=shape.name,
        graph="block",
        backend="hybrid_router_ane",
        compute_unit="CPU_AND_NE",
        actual_execution="MIXED",
        batch=batch,
        precision="fp16",
        timing=t,
        boundary_us=null.get("boundary_us", 0.0),
        dispatch_us=null.get("dispatch_us", 0.0),
        kernel_us=t.median_ms * 1000.0,
        coreml_warnings_log=str(log.relative_to(REPO_ROOT)),
        notes="router on CoreML (compute_unit hint=NE), expert on MLX. "
              "actual_execution requires Instruments to confirm router placement.",
    )


def placement_C_expert_coreml(shape: BlockShape, batch: int, iters: int = 200) -> ResultRow:
    """Router on MLX, expert on CoreML. Pays one boundary round-trip per call."""
    log = RESULTS_DIR / f"block_expert_{shape.name}_b{batch}.coreml_warnings.log"
    pkg = _convert(_ExpertANE(shape.hidden, shape.ffn),
                   f"block_expert_{shape.name}", batch, shape.hidden, log)
    expert = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)

    W_gate, _, _, _ = _make_mlx_weights(shape)
    x = mx.array(np.random.randn(batch, shape.hidden).astype(np.float16))
    mx.eval(x)

    def fwd():
        logits = mx.softmax(x @ W_gate, axis=-1)
        w_mx = logits[:, :1]
        x_np = np.asarray(x, dtype=np.float16).reshape(batch, shape.hidden, 1, 1)
        e_out = expert.predict({"x": x_np})["y"]  # (B, hidden, 1, 1)
        e_mx = mx.array(e_out.reshape(batch, shape.hidden))
        out = e_mx * w_mx
        mx.eval(out)
        return out

    fwd()

    def step():
        fwd()

    t = benchmark(step, iters=iters, warmup=20)
    null = load_null(batch) or {}
    return ResultRow(
        shape=shape.name,
        graph="block",
        backend="hybrid_expert_ane",
        compute_unit="CPU_AND_NE",
        actual_execution="MIXED",
        batch=batch,
        precision="fp16",
        timing=t,
        boundary_us=null.get("boundary_us", 0.0),
        dispatch_us=null.get("dispatch_us", 0.0),
        kernel_us=t.median_ms * 1000.0,
        coreml_warnings_log=str(log.relative_to(REPO_ROOT)),
        notes="router on MLX, expert on CoreML (compute_unit hint=NE). "
              "actual_execution requires Instruments to confirm expert placement.",
    )


def placement_D_all_coreml(shape: BlockShape, batch: int, iters: int = 200) -> ResultRow:
    """Whole block as one CoreML graph. No per-call MLX boundary."""
    log = RESULTS_DIR / f"block_all_{shape.name}_b{batch}.coreml_warnings.log"
    pkg = _convert(_BlockANE(shape.hidden, shape.ffn, shape.experts),
                   f"block_all_{shape.name}", batch, shape.hidden, log)
    model = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)

    x_np = np.random.randn(batch, shape.hidden, 1, 1).astype(np.float16)
    model.predict({"x": x_np})

    def step():
        model.predict({"x": x_np})

    t = benchmark(step, iters=iters, warmup=20)
    null = load_null(batch) or {}
    boundary_us = null.get("boundary_us", 0.0)
    dispatch_us = null.get("dispatch_us", 0.0)
    return ResultRow(
        shape=shape.name,
        graph="block",
        backend="hybrid_all_coreml",
        compute_unit="CPU_AND_NE",
        actual_execution="UNKNOWN",
        batch=batch,
        precision="fp16",
        timing=t,
        boundary_us=boundary_us,
        dispatch_us=dispatch_us,
        kernel_us=max(0.0, t.median_ms * 1000.0 - boundary_us - dispatch_us),
        coreml_warnings_log=str(log.relative_to(REPO_ROOT)),
        notes="single CoreML graph for the whole block. No MLX. "
              "actual_execution requires Instruments to confirm placement.",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


PLACEMENTS = {
    "A": ("all_mlx", placement_A_all_mlx),
    "B": ("router_ane", placement_B_router_coreml),
    "C": ("expert_ane", placement_C_expert_coreml),
    "D": ("all_coreml", placement_D_all_coreml),
}


def main():
    if len(sys.argv) >= 3:
        shapes = [block_by_name(sys.argv[1])]
        batches = [int(sys.argv[2])]
        which = sys.argv[3] if len(sys.argv) >= 4 else "ABCD"
    else:
        shapes = BLOCK_SHAPES
        batches = [1, 8, 32]   # subset; energy run can do more
        which = "ABCD"

    for shape in shapes:
        for b in batches:
            for k in which:
                if k not in PLACEMENTS:
                    continue
                label, fn = PLACEMENTS[k]
                row = fn(shape, b)
                row.write()
                tag = f"[{k}] {label} block/{shape.name} B={b}"
                print(f"  {tag:<48} median={row.timing.median_ms:>9.3f}ms  "
                      f"p95={row.timing.p95_ms:>9.3f}ms  "
                      f"first={row.timing.first_call_ms:.3f}ms")


if __name__ == "__main__":
    main()
