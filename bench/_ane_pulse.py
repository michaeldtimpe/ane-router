"""Tight-loop CoreML pulser for ANE rail verification.

Builds a small linear+softmax model targeting CPU_AND_NE, runs it in a tight
loop until killed. Used by scripts/verify_ane.sh — the bash side runs
powermetrics during the pulse and checks the ANE rail.

Run with:
    python -m bench._ane_pulse [hidden] [seconds]
Defaults: hidden=4096, seconds=8 (loop runs for `seconds`, then exits).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_model(hidden: int):
    import coremltools as ct
    import torch
    import torch.nn as nn

    cache = MODELS_DIR / f"pulse_h{hidden}.mlpackage"
    if cache.exists():
        return ct.models.MLModel(str(cache), compute_units=ct.ComputeUnit.CPU_AND_NE)

    class Pulse(nn.Module):
        # ANE-canonical form: input (B, C_in, 1, S), Conv2d 1x1 acts as Linear.
        def __init__(self, h: int):
            super().__init__()
            self.fc = nn.Conv2d(h, h, kernel_size=1, bias=False)

        def forward(self, x):
            return torch.softmax(self.fc(x), dim=1)

    ex = torch.randn(1, hidden, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(Pulse(hidden).eval().half(), ex)
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
    return ct.models.MLModel(str(cache), compute_units=ct.ComputeUnit.CPU_AND_NE)


def main():
    hidden = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    model = build_model(hidden)
    x = np.random.randn(1, hidden, 1, 1).astype(np.float16)
    print(f"pulse: hidden={hidden} seconds={seconds}", flush=True)
    deadline = time.monotonic() + seconds
    n = 0
    while time.monotonic() < deadline:
        model.predict({"x": x})
        n += 1
    print(f"pulse: {n} iters in {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
