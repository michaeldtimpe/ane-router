#!/usr/bin/env bash
# Resume the router CoreML sweep from the cells that didn't complete.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv-312/bin/python"
export PYTHONPATH="."

echo "=== resume: stress B=128 (CPU_AND_NE, ALL) ==="
"$PY" bench/router_coreml.py stress 128 CPU_AND_NE 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version|E5RT)"
"$PY" bench/router_coreml.py stress 128 ALL 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version|E5RT)"

echo
echo "=== resume: stress B=512 ==="
for u in CPU_ONLY CPU_AND_GPU CPU_AND_NE ALL; do
    "$PY" bench/router_coreml.py stress 512 "$u" 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version|E5RT)"
done

echo
echo "=== resume: router_stress_dense (all batches) ==="
for b in 1 2 4 8 32 128 512; do
    for u in CPU_ONLY CPU_AND_GPU CPU_AND_NE ALL; do
        "$PY" bench/router_coreml.py router_stress_dense "$b" "$u" 2>&1 | grep -vE "^(Running MIL|Converting PyTorch|Torch version|E5RT)"
    done
done

echo
echo "=== resume done ==="
