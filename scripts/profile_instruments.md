# Instruments Core ML profiling — manual workflow

The Core ML compiler decides which ops run on CPU vs GPU vs ANE. The
`compute_units=CPU_AND_NE` argument is a hint, not a guarantee.
Instruments is the only ground-truth way to find out where each op landed.

This is a manual GUI workflow; there is no scriptable equivalent for
the Core ML template.

## One-time: install

Xcode is required (not just CLT). After install:

```
xcrun --find instruments
```

If that returns a path, you're set.

## Profile a single CoreML benchmark

1. Pick a benchmark you want to profile, e.g.
   `bench/router_coreml.py mixtral-8x7b 1 CPU_AND_NE`. Edit the script to
   loop forever (or sleep before exit) so Instruments can attach.
   Quickest way: at the bottom of `bench_one`, replace the timed loop
   with `while True: model.predict({"x": x})`.
2. Start the benchmark in one terminal.
3. In another terminal: `open -a Instruments` (or launch from Xcode).
4. From the template chooser: pick **Core ML**.
5. Set the target to the running `python` process. (Top-left dropdown.)
6. Click record.
7. Let it record 5–10 seconds. Stop.

## Read the trace

The Core ML template has multiple tracks:

- **Core ML Models** — shows each `predict()` call as a span. Click a span
  to see per-op timing.
- **Neural Engine** — shows ANE activity. Compare to the Core ML Models
  track: any `predict()` span with no overlapping ANE activity ran on
  CPU/GPU.
- **GPU** — shows Metal command buffers for any GPU work.

For a single converted model, look at one `predict()` span. The "Compute
Unit" attribute on each op (in the bottom detail panel) tells you
where it ran. Possible values: `CPU`, `GPU`, `Neural Engine`.

## Map to `actual_execution`

For a result JSON that was produced with `compute_unit=CPU_AND_NE`:

| Instruments shows                                     | actual_execution |
|-------------------------------------------------------|------------------|
| All ops ran on Neural Engine                          | `ANE`            |
| All ops ran on GPU                                    | `GPU`            |
| All ops ran on CPU                                    | `CPU`            |
| Mix (some on ANE, some on CPU/GPU)                    | `MIXED`          |
| Requested `CPU_AND_NE` but everything ran on GPU/CPU  | `ANE_FALLBACK`   |

## Export

File → Export Trace → save to `results/instruments/<shape>_<batch>_<unit>.trace`.

Take a screenshot of the per-op partition view; save next to the
`.trace`. The `.trace` is openable later in Instruments; the screenshot
is for the writeup.

## Patch the result JSONs

Once you know `actual_execution` for a (shape, batch, compute_unit)
cell, run:

```
.venv-312/bin/python scripts/patch_actual_execution.py \
    --shape mixtral-8x7b --batch 1 --unit CPU_AND_NE \
    --actual ANE_FALLBACK --trace results/instruments/<file>.trace
```

This walks `results/raw/*.json` and patches every matching row.
