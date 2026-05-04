# ane-router

Microbenchmark answering whether running an MoE LLM's router and/or expert
MLPs on the Apple Neural Engine (ANE) is worthwhile vs Metal/GPU on an
M1 Max.

**Headline answer:**
- ANE has no path through MLX (MLX is Metal-only).
- Inside CoreML, ANE on M1 Max is only a winner for *power savings at the
  cost of performance*. It loses on latency to MLX everywhere; it wins
  37–63% on energy per token for expert-MLP workloads at sustained load.
- Routers are too small to benefit; keep them on CPU.

See [FINDINGS.md](FINDINGS.md) for the full writeup with measured numbers
and [results/SUMMARY_MATRIX.md](results/SUMMARY_MATRIX.md) for the raw
benchmark grid.

## The three nested questions

1. Is ANE faster for isolated ops? (kernel-only)
2. Is ANE more energy-efficient for sustained workloads? (mJ/token)
3. **Can ANE coexist efficiently with GPU in one pipeline?** ← the practical
   question. Crossing frameworks kills you. Boundary cost is a first-class
   metric here, not a side note.

## Falsifiable hypotheses

- **H1** Mixtral-shaped router `Linear(4096, 8)` at B=1: ANE loses on latency
  by 5-10x vs GPU. Hybrid (GPU stub -> ANE router -> GPU stub) loses to
  pure-GPU by >=1 ms from boundary crossings.
- **H2** Mixtral-shaped expert MLP `Linear(4096, 14336) + Linear(14336, 4096)`
  at B>=32: ANE is competitive on latency and likely wins on mJ/token at
  sustained throughput.
- **H3** Full MoE block at decode (B=1): ANE may win on energy at sustained
  throughput but loses on per-token latency vs MLX-only. The deployment story
  is energy savings, not latency.
- **H4** A crossover shape exists between Mixtral-router and Mixtral-expert
  where ANE starts being competitive. The synthetic dense `Linear(4096, 4096)`
  router-stress shape bisects this gap.

## Setup

Python 3.14 currently fails because coremltools' `libcoremlpython` runtime is
not built for 3.14, so models can be converted but not executed in Python.
Use Python 3.12.

```
/opt/homebrew/bin/python3.12 -m venv .venv-312
.venv-312/bin/pip install -r requirements.txt
```

`ane-transformers` is intentionally NOT in requirements: it pins
`protobuf==3.20.1` which conflicts with newer coremltools. The verification
script wakes the ANE with a tiny coremltools-built model directly.

## Running

Energy measurements require sudo (powermetrics). Pre-authenticate:

```
sudo -v
```

Order:

```
.venv-312/bin/python bench/null_experiments.py        # boundary + dispatch floor
sudo .venv-312/bin/python bench/null_experiments.py --energy-baseline   # idle wattage
scripts/verify_ane.sh                                 # gate: must pass
.venv-312/bin/python bench/router_torch_cpu.py
.venv-312/bin/python bench/router_mlx.py
.venv-312/bin/python bench/router_coreml.py
.venv-312/bin/python bench/block_hybrid.py            # the experiment
.venv-312/bin/python analyze.py
```

## Known limitations

**`dispatch_us` calibration is symmetric.** The null `coreml_empty` model has
input and output both shaped `(B, 4096, 1, 1)`. CoreML python-bridge cost
scales with both input and output size, so for asymmetric models (router has
4096 in, 8 out) the calibrated `dispatch_us` *overestimates* the real
boundary cost. The recorded `kernel_us` will floor at 0 in that case. The
total median latency is correct; only the kernel/dispatch breakdown is loose
for now. Per-shape calibration is a TODO.

**`actual_execution` defaults to UNKNOWN for CoreML rows.** The compiler may
ignore the `compute_units` request. Run Instruments per shape (see
`scripts/profile_instruments.md`), then patch with
`scripts/patch_actual_execution.py`.

**Energy data requires sudo.** Run `sudo -v` before `verify_ane.sh` or any
energy benchmark.

## Pitfalls (read before trusting any number)

1. `compute_units=CPU_AND_NE` is a hint. Verify with Instruments per
   converted model. Each result row carries `actual_execution`
   (`ANE | GPU | CPU | MIXED | ANE_FALLBACK`) derived from the trace, not
   from the request.
2. Any fp32 op forces CPU fallback. Set `compute_precision=FLOAT16`
   at convert time. Conversion warnings captured to
   `results/raw/<run>.coreml_warnings.log`.
3. ANE prefers `(B, C, 1, S)` channels-first 4D layouts.
4. First-call penalty: cache `.mlpackage` to disk; warm >=100 calls.
5. Unified memory is not free transfer in Python: coremltools wrappers
   memcpy through numpy. True zero-copy needs Swift + IOSurface-backed
   `MLMultiArray`. **Python benchmark is pessimistic for hybrid.**
6. `asitop` ANE column is unreliable for quantitative claims. Use
   `powermetrics -i 100` directly.
7. `top_k` with runtime k often forces CPU fallback.
8. Thermal throttling: AC, lid open, cool ambient. Reject runs with
   non-nominal `pmset -g thermlog`.
9. Energy: idle baseline subtracted, first 5 s discarded, per-device
   sanity check (ANE rail must dominate for ANE rows).
10. "32 MB of ANE memory" is folklore. ANE has weight cache (~6-16 MB
    estimates) but all data is in unified DRAM. Constraint is dispatch
    latency, not capacity.
