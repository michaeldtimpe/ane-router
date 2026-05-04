#!/usr/bin/env bash
# Run the full expert MLP sweep across all backends, all shapes, all batches.
# Backends are sequential (CPU/GPU contention).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv-312/bin/python"
export PYTHONPATH="."

echo "=== torch_cpu expert sweep ==="
"$PY" bench/expert_torch_cpu.py 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version)"

echo
echo "=== mlx expert sweep ==="
"$PY" bench/expert_mlx.py 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version)"

echo
echo "=== coreml expert sweep (all 4 ComputeUnits) ==="
"$PY" bench/expert_coreml.py 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version|E5RT|^[[:space:]]*$)"

echo
echo "=== expert sweep done ==="
