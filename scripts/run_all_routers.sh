#!/usr/bin/env bash
# Run all router benchmarks across all shapes x all batches.
# Backends are run sequentially so they don't interfere with each other.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv-312/bin/python"
export PYTHONPATH="."

echo "=== torch_cpu router sweep ==="
"$PY" bench/router_torch_cpu.py 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version)"

echo
echo "=== mlx router sweep ==="
"$PY" bench/router_mlx.py 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version)"

echo
echo "=== coreml router sweep (all 4 ComputeUnits) ==="
"$PY" bench/router_coreml.py 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version|E5RT|^[[:space:]]*$)"

echo
echo "=== full router sweep done ==="
