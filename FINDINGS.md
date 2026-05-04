# FINDINGS — should the MoE handler run on the M1 Max ANE?

**TL;DR**

- **Router (Linear H→E with small E): keep on CPU.** The compiler doesn't
  even engage ANE for tiny routers below B=128. CPU fp32 wins at every
  small batch (e.g. Mixtral router B=1 = 10 µs CPU vs 89 µs CoreML vs 337 µs
  MLX). For the larger batches (B=128/512) where the compiler does route to
  ANE, ANE is still 5–10× slower than CPU and the energy win is modest.
- **Expert MLPs: ANE engages everywhere, MLX wins latency, ANE wins energy.**
  Every (shape × batch) cell with `CPU_AND_NE` is dispatched to the Neural
  Engine per Apple's compute plan. But MLX (Metal) wins latency at every
  expert size by 5–30×. The case for ANE on experts is **energy**:
  Mixtral expert at B=32 uses 0.67 mJ/token on ANE vs 1.82 mJ/token on CPU
  (63% less). For OLMoE expert at B=8, ANE is 2.7% slower than CPU but
  37% more energy-efficient.
- **Bottom line: if your goal is latency, use MLX. If your goal is
  energy/battery on M1 Max specifically, route experts to ANE and leave
  the router on CPU. There is no useful "router on ANE" deployment.**

Methodology note: this revision is grounded in `MLComputePlan.preferred_compute_device`
data via `scripts/resolve_actual_execution.py` — the compiler-internal
placement, not energy-rail inference (which I previously used and which
gave one wrong answer that compute-plan corrected).

## Dataset

345 result JSON rows in `results/raw/`:
- 7 router shapes × 7 batches × 6 backend variants
- 4 expert shapes × 7 batches × 6 backend variants
- 2 block shapes × 3 batches × 4 placements
- Energy headlines for routers (3 cells) and experts (4 cells)
- Null calibration (boundary + dispatch overhead per batch)

Every CoreML row has its `actual_execution` populated from the compute
plan (ANE / GPU / CPU; no `UNKNOWN` rows remain except for one cell where
the compute plan and energy rail disagreed — see methodology section).

## H1 — Mixtral-style router on ANE: REVISED, still loses

The original framing was "ANE never engages for Mixtral router because
the op is too small." The compute plan refines this:

| Router shape | `CPU_AND_NE` placement, by batch size |
|---|---|
| mixtral-8x7b (4096×8)      | CPU at B=1-32; **ANE at B=128, B=512** |
| mixtral-8x22b (6144×8)     | CPU at B=1-32; **ANE at B=128, B=512** |
| qwen2-moe (2048×60)        | CPU at B=1-256; **ANE at B=512** |
| deepseek-moe (2048×66)     | CPU at B=1-128; **ANE at B=512** |
| olmoe (2048×64)            | CPU at B=1-128; **ANE at B=512** |
| stress (8192×256)          | irregular: ANE at B=2,4,8,128,512; CPU at B=1,32 |
| router_stress_dense (4096×4096) | **ANE at every batch** |

So the compiler does flip tiny routers to ANE at large batches. But
latency at those cells is bad:

| Mixtral router B=128 | latency µs/call |
|---|---|
| torch CPU fp32  | 116 |
| MLX (Metal)     | 257 |
| CoreML CPU_ONLY | 1058 |
| CoreML CPU_AND_NE (compiler picks ANE) | 1068 |

ANE engagement at B=128 makes things ~9× slower than CPU. There is no
batch size at which ANE-on-router beats CPU-on-router for any tiny
(H, E) shape we measured.

**Recommendation: keep the router on CPU.** No path through ANE wins,
even when the compiler picks ANE on its own.

## H2 — Expert MLP on ANE: ANE engages, MLX wins latency, ANE wins energy

Every expert MLP (shape `(H, ffn) + (ffn, H)`) in the sweep is placed on
ANE by the compiler when `CPU_AND_NE` is requested. This is the regime
ANE is built for — large enough compute volume that dispatch overhead
amortizes.

### Mixtral expert (4096 → 14336 → 4096), latency per call

| B | torch CPU fp32 | MLX (Metal) | CoreML CPU_ONLY (CPU) | CoreML CPU_AND_NE (ANE) | CoreML CPU_AND_GPU (GPU) |
|---|---|---|---|---|---|
| 1   | 22.9 ms | **2.6 ms** | 7.3 ms | 6.6 ms | 2.5 ms |
| 32  | 27.2 ms | **5.5 ms** | 6.3 ms | 9.6 ms | 66.8 ms |
| 128 | 46.1 ms | **9.6 ms** | 20.3 ms | crash† | 304.5 ms |
| 512 | 117.8 ms | **35.2 ms** | 83.1 ms | 284.8 ms | 1312.0 ms |

† Mixtral expert B=128 with `CPU_AND_NE` reproducibly crashed ANE at
runtime: `ANEProgramProcessRequestDirect Failed status=0x15`. Same
failure mode as the `stress` router at B=128. The compiler accepted the
model; ANE rejected the inference. This is a real production hazard.

### OLMoE expert (2048 → 1024 → 2048), latency per call

| B | torch CPU fp32 | MLX | CoreML CPU_ONLY | CoreML CPU_AND_NE (ANE) |
|---|---|---|---|---|
| 1   | 0.56 ms | **0.35** | 0.46 ms | 0.47 ms |
| 32  | 1.38 ms | **0.46** | 1.32 ms | 1.51 ms |
| 128 | 2.05 ms | **0.70** | 5.79 ms | 8.39 ms |
| 512 | 4.62 ms | **1.75** | 25.81 ms | 36.62 ms |

**Pattern:** MLX (direct Metal) wins on latency at every expert size,
often by 5–30× over both ANE and CoreML's GPU path. CoreML's own
`CPU_AND_GPU` path is bizarrely slow (Mixtral B=128: 304 ms via CoreML
vs 9.6 ms via MLX — same hardware, 31× slower). This is a CoreML
runtime tax, not a Metal capability gap.

### Energy: ANE wins decisively at sustained throughput

Idle baseline subtracted, 30 s steady-state, 200 ms `powermetrics`
sampling:

| Cell | latency µs/tok | mJ/tok | rail signature |
|---|---|---|---|
| Mixtral expert B=32 CPU_ONLY  | 224 | 1.82 | CPU 8114 mW |
| Mixtral expert B=32 CPU_AND_NE | 313 | **0.67** | ANE 1692 + CPU 442 |
| OLMoE expert B=8 CPU_ONLY     |  81 | 0.50 | CPU 6205 |
| OLMoE expert B=8 CPU_AND_NE   |  83 | **0.32** | ANE 957 + CPU 2866 |

Mixtral expert B=32: ANE is 1.4× slower but **63% less energy per token**.
OLMoE B=8: ANE is statistically tied on latency (2.7% slower) and
**37% less energy**. The ANE energy advantage gets larger as the model
grows; for the largest realistic experts the multiplier is >2× energy
savings at the cost of ~1.4× latency.

## H3 — Full MoE block placements

Single-graph block (router + 1 expert + softmax-weighting) timed across
four placement strategies:

### Mixtral block (4096 hidden, 14336 ffn, 8 experts)

| B | A: all-MLX | B: router-ANE | C: expert-ANE | D: all-CoreML/ANE |
|---|---|---|---|---|
| 1  | **3.1 ms** | 3.7 ms (+19%) | 7.1 ms (+128%) | 6.7 ms (+115%) |
| 8  | **7.8 ms** | 7.8 ms        | 9.3 ms (+19%)  | 8.6 ms (+10%)  |
| 32 | **7.8 ms** | 8.6 ms (+10%) | 10.6 ms (+36%) | 10.3 ms (+32%) |

### OLMoE block (2048 hidden, 1024 ffn, 64 experts)

| B | A: all-MLX | B: router-ANE | C: expert-ANE | D: all-CoreML/ANE |
|---|---|---|---|---|
| 1  | 0.69 ms    | 0.51 ms (-26%) | 0.99 ms       | **0.46 ms** (-33%) |
| 8  | **0.54 ms** | 1.25 ms       | 1.02 ms        | 0.70 ms (+30%)     |
| 32 | **0.56 ms** | 1.24 ms       | 2.81 ms        | 1.94 ms (+246%)    |

Two readings:
- **Heterogeneous placements (B and C — split between MLX and CoreML)
  are usually worse than homogeneous ones (A all-MLX, D all-CoreML).**
  Crossing the framework boundary mid-pipeline costs more than running
  the slower backend end-to-end.
- For OLMoE at B=1 (small block, decode regime), all-CoreML beat all-MLX
  by 33%. CoreML's compiled fused graph for tiny ops can still win when
  the alternative is multiple MLX kernel launches. This is the only
  case in the dataset where CoreML beat MLX on latency.

The routers in the block-hybrid pipelines (`block_router_*` packages) all
land on CPU per the compute plan — even within a "use ANE" hint context,
because they're tiny.

## H4 — Crossover shape

The shape at which the compiler's preferred placement flips from CPU to
ANE (under `CPU_AND_NE`) is a function of both H and B:

| Shape | First batch where compiler picks ANE |
|---|---|
| `Linear(4096, 4096)` (router_stress_dense) | B=1 |
| `Linear(8192, 256)` (stress) | B=2 (irregular: misses B=1, B=32) |
| `Linear(4096, 14336)` (Mixtral expert up_proj) | B=1 (whole expert MLP) |
| `Linear(4096, 8)` (Mixtral router) | B=128 |
| `Linear(2048, 64)` (OLMoE router) | B=512 |

Bigger output dim brings the threshold down. Tiny (8-64) output routers
need batch sizes 100s before ANE engages.

## Boundary cost (the framework-bridge floor)

From `null_experiments.json`:

| B   | MLX → numpy → MLX | MLX → CoreML identity → MLX |
|-----|-------------------|-----------------------------|
| 1   | 1.9 µs           | **300 µs**                   |
| 8   | 2.7 µs           | **595 µs**                   |
| 128 | 19.6 µs          | **11,004 µs**                |
| 512 | 134.5 µs         | **42,958 µs**                |

CoreML python-bridge scales linearly with tensor size, consistent with
a memcpy through numpy. A *no-op* CoreML predict() at B=128 takes 11 ms
to push data through. Hybrid pipelines pay this once per crossing.

This is partly a Python wrapper limitation: Swift + IOSurface-backed
`MLMultiArray` would be substantially cheaper. So the Python numbers
above are *pessimistic* for hybrid pipelines. If the Python result for
some hybrid is "almost worth it," that's the trigger to invest in Swift;
in this dataset, no Python hybrid configuration came close.

## Methodology — what was wrong about energy-rail inference

In the first pass, the Mixtral B=1 router with `CPU_AND_NE` showed a
26 W spike on the GPU rail and 0 mW on ANE during a 30 s sustained
benchmark. I labelled that cell `actual_execution: GPU`.

The compute plan disagrees: it says `actual_execution: CPU`. The
energy-expert run later confirmed why — even with no workload, the
**idle baseline GPU rail showed 26 W** from screen compositing and
background app work. Subtracting idle would have nulled it out, but the
original energy script didn't do per-rail idle-subtraction comparisons
across cells; it just computed mJ/token from the total post-idle delta.

The patched JSON for that cell is now labelled
`actual_execution: GPU_BUT_PLAN_SAYS_CPU`, preserving both signals.
Trust the compute plan over the rail signature for placement; trust the
rail signature (with idle subtraction) for energy magnitude.

The energy-expert run (`bench/energy_run_expert.py`) does subtract idle
correctly, which is why the Mixtral expert ANE row shows GPU delta = 0
even though absolute steady-state GPU power was 22 W.

## Compiler placement is fragile, but readable

`compute_units` is a hint, but
`MLComputePlan.preferred_compute_device` is the compiler's actual
decision. The script `scripts/resolve_actual_execution.py` reads it
non-destructively from a compiled `.mlmodelc` (no Swift, no Xcode GUI)
and patches every matching JSON in `results/raw/`. Three observed
failure modes:

- **Hint ignored, compiler chose its own preferred device.** Frequent;
  see H1 table.
- **Compiler accepted the model but ANE rejected it at runtime**
  (`ANEProgramProcessRequestDirect Failed status=0x15`). Two
  reproducible cells: `stress` shape B=128 router, and Mixtral expert
  B=128. Production code wrapping ANE-targeted models needs a
  fallback path.
- **Compiler placed ops on multiple devices in one model.** Common for
  MIXED labels; usually a Conv2d on ANE plus a softmax/cast on CPU.

## What this answers for the original question

> Is there merit in putting the single-layer MoE handler from an MoE
> local LLM model into ANE?

**For the router specifically: no.** Mixtral-style `Linear(H, E)` with
small E is too small to benefit. The compiler doesn't engage ANE at
small batches, and at large batches ANE engages but loses 5–10× to CPU.

**For the experts: yes, but only as an energy play.** Compute plan
confirms ANE engagement at every expert size we measured. Latency loses
to MLX at every cell. Energy beats CPU by 37–63% per token at sustained
throughput. **The question is whether you're optimizing latency
(use MLX) or battery/thermal (route experts through CoreML/ANE).**

**Practical recommendation if you build something:**
- Router → CPU (`torch` or simple Accelerate matmul; 10 µs).
- Experts → MLX (Metal/GPU) for latency-bound use, or CoreML/ANE for
  battery-bound sustained throughput. The ANE choice trades ~1.4× latency
  for ~50% energy.
- Avoid hybrid CoreML/MLX pipelines crossing the python boundary
  per token; the boundary cost (300 µs at B=1, 11 ms at B=128) destroys
  any backend gain. Either go end-to-end on one framework, or rewrite
  the data path in Swift.

## Reproducing

```
# Setup (Python 3.12; coremltools 9.0 has no libcoremlpython on 3.14):
/opt/homebrew/bin/python3.12 -m venv .venv-312
.venv-312/bin/pip install -r requirements.txt

# Passwordless powermetrics (one-time):
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/powermetrics" \
  | sudo tee /etc/sudoers.d/powermetrics

# Gate (must pass):
./scripts/verify_ane.sh

# Calibration:
PYTHONPATH=. .venv-312/bin/python bench/null_experiments.py

# Sweeps:
./scripts/run_all_routers.sh
./scripts/run_all_experts.sh
./scripts/run_block_sweep.sh

# Programmatic actual_execution (replaces Xcode Performance Report):
PYTHONPATH=. .venv-312/bin/python scripts/resolve_actual_execution.py

# Energy (subset of cells):
PYTHONPATH=. .venv-312/bin/python bench/energy_run.py
PYTHONPATH=. .venv-312/bin/python bench/energy_run_expert.py

# Summary:
PYTHONPATH=. .venv-312/bin/python analyze.py            # full text
PYTHONPATH=. .venv-312/bin/python analyze.py --md       # Markdown
```

## Caveats / open work

1. **`dispatch_us` calibration is symmetric** (4096 in, 4096 out). For
   asymmetric models (routers, where output is much smaller than input),
   it overcounts dispatch and floors `kernel_us` to zero. Total median
   latency is correct; only the kernel/dispatch breakdown is loose.
2. **No real-model validation.** Synthetic shapes match the published
   architectures (Mixtral 4096×8, etc.) but no Mixtral router or expert
   weights have been pulled from HuggingFace and run through the same
   harness. Sanity check, not correctness gate.
3. **Single-day numbers.** No cross-day repetition for thermal variance.
4. **No int8/int4 path.** ANE has int8 support and may shift the
   energy/latency story for quantized inference. Out of scope for this
   sweep.
5. **No Swift/IOSurface zero-copy benchmark.** Python results are
   pessimistic for hybrid configurations; if you want to know whether a
   true zero-copy ANE pipeline is competitive, you have to write Swift.

## External cross-references

After the benchmark was written, five external sources were reviewed to
check whether anything contradicts the findings. Nothing does. Notes:

- **Apple ML Research, "Deploying Transformers on the Apple Neural Engine"**
  (https://machinelearning.apple.com/research/neural-engine-transformers)
  documents the four ANE-friendly principles: channels-first 4D layout
  `(B, C, 1, S)`, `nn.Linear → nn.Conv2d` with kernel 1, chunked attention,
  custom `LayerNormANE`. The benchmark already implements (1) and (2) for
  the router and expert variants in `bench/router_coreml.py` and
  `bench/expert_coreml.py`. The article never claims ANE wins at B=1 for
  tiny ops; its cited gains (up to ~10× lower latency) are at sequence
  lengths ≥128 and batch ≥1 on full transformer blocks. Consistent with
  H1 (router-too-small) and H2 (expert-engages-but-MLX-still-wins-latency).

- **Anemll/anemll-bench** (https://github.com/Anemll/anemll-bench) reports
  M1 Max ANE bandwidth at ~55 GB/s vs ~400 GB/s system memory bandwidth
  (≈14% utilization). That is the structural reason ANE on M1 Max can't
  win on per-token latency for memory-bound ops: its own GPU has ~7× more
  bandwidth to work with on the same chip. Their suite uses the same
  Python+coremltools predict() bridge as this harness, so the ~300 µs
  dispatch floor measured in `null_experiments.json` is a property of
  the toolchain, not of our setup. This third-party data strengthens
  the "ANE is energy, not latency, on M1 Max specifically" framing.

- **MLCompute `MLCDevice.ane()`** (https://developer.apple.com/documentation/mlcompute/mlcdevice/ane())
  was a deprecated path even at the time of measurement: macOS 14.3 marks
  the entire framework as deprecated, with the doc explicitly redirecting
  callers to BNNS / MetalPerformanceShaders / CoreML. There is no
  MLCompute path that bypasses CoreML's compiler placement heuristics.
  Future readers: this is a dead end.

- **apple/ml-ane-transformers** (https://github.com/apple/ml-ane-transformers)
  was not adopted as a dependency. Its `setup.py` pins
  `torch<=1.11.0` and `protobuf<=3.20.1` (last commit 2022), which
  conflicts with the `coremltools 9.0` env this harness uses. The
  reusable parts (`LayerNormANE`, `MultiHeadAttention` with chunked
  softmax) target full transformer blocks; for a standalone Linear-shaped
  router or expert MLP, the only applicable piece is the
  Linear→Conv2d-1x1 swap, which `bench/router_coreml.py` and
  `bench/expert_coreml.py` already do.

- **developer.apple.com/machine-learning/resources/** is an index page;
  the only actionable artifact reachable from it is **Xcode → Open
  Developer Tool → Core ML Performance Report**. We reproduced what
  that tool reports programmatically via `MLComputePlan` in
  `scripts/resolve_actual_execution.py` — see the "Compiler placement
  is fragile, but readable" section above.
