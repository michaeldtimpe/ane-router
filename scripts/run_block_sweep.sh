#!/usr/bin/env bash
# Run block_hybrid for both BLOCK_SHAPES at B in {1, 8, 32}, all 4 placements.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv-312/bin/python"
export PYTHONPATH="."

echo "=== block_hybrid sweep ==="
for shape in mixtral-8x7b olmoe; do
    for b in 1 8 32; do
        for plc in A B C D; do
            "$PY" bench/block_hybrid.py "$shape" "$b" "$plc" 2>&1 \
                | grep -vE "^(Running MIL|Converting PyTorch|Torch version|E5RT)"
        done
    done
done
echo
echo "=== block sweep done ==="
