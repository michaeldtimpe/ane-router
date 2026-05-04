# Xcode Core ML Performance Report — manual workflow

The Performance Report is the cheapest way to confirm where the CoreML
compiler placed each op (CPU / GPU / Neural Engine). It runs the model
on this Mac and reports per-op compute-unit assignment plus median latency,
without writing any Swift code or attaching Instruments.

## One-time

Xcode is required (not just Command Line Tools).

## Per-model workflow

1. Open Xcode.
2. Menu: **Xcode → Open Developer Tool → Create ML**? — wrong tool. Use:
   **File → Open…** and select an `.mlpackage` directly.
3. With the model open, click the **Performance** tab at the top.
4. Click the **+** button to add a Performance Report. Pick **This Mac**
   (compute units: All) and confirm.
5. Xcode runs the model end-to-end and shows:
   - Median prediction time (ms).
   - **Compute Unit Mapping** with a row per layer:
     `CPU` | `GPU` | `Neural Engine`. This is the ground truth for
     `actual_execution`.
   - Load time (cold start cost).

## Models worth profiling

In priority order:

- `models/router_mixtral-8x7b_b1.mlpackage` — energy data inferred
  `actual_execution: GPU` (26 W rail spike during workload). Performance
  Report should show every op on GPU. If it shows Neural Engine instead,
  the energy interpretation needs revising.
- `models/router_router_stress_dense_b128.mlpackage` — energy data
  inferred `MIXED` (CPU 1788 mW + ANE 603 mW). Performance Report should
  show some ops on Neural Engine and others on CPU.
- `models/router_stress_b128.mlpackage` — this one crashed ANE at runtime
  (`ANEProgramProcessRequestDirect Failed status=0x15`). Performance
  Report may surface *why* the compiler accepted the model but ANE
  rejected the inference: an op on the Neural Engine row that's flagged,
  or an unsupported tensor shape.

Optional extras: any of `router_*_b128.mlpackage` for the named MoE
families (mixtral-8x22b, qwen2-moe, deepseek-moe, olmoe).

## Patching results back

For each (shape, batch, ComputeUnit) combination the report resolves, run:

```
PYTHONPATH=. .venv-312/bin/python scripts/patch_actual_execution.py \
    --shape <shape> --graph router_only --batch <B> \
    --unit CPU_AND_NE --actual <ANE|GPU|CPU|MIXED|ANE_FALLBACK>
```

The patch helper updates every matching JSON in `results/raw/`.
