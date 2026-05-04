"""Read per-op compute-unit assignment for every models/*.mlpackage and
patch matching result JSONs in results/raw/.

Replaces the Xcode-Performance-Report manual workflow with a scriptable
one: coremltools.models.compute_plan.MLComputePlan.load_from_path returns
the same data the Xcode Performance Report displays, including which
ops landed on Neural Engine vs GPU vs CPU.

Usage:
    PYTHONPATH=. .venv-312/bin/python scripts/resolve_actual_execution.py
        [--shape <name>] [--graph <kind>] [--dry-run]

Without arguments: walks every models/*.mlpackage, derives per-package
placement, patches every results/raw/*.json that matches.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results" / "raw"

UNIT_LABEL = {
    "MLComputeDeviceTypeNeuralEngine": "ANE",
    "MLComputeDeviceTypeGPU": "GPU",
    "MLComputeDeviceTypeCPU": "CPU",
}


def _walk_ops(model_struct) -> Iterable:
    """Yield ML Program operations from the model structure tree.

    coremltools' MLModelStructure has a `program` attribute when the model
    is an ML Program. The program has functions, each function has a
    block, each block has operations (which may themselves contain
    nested blocks for control flow).
    """
    program = getattr(model_struct, "program", None)
    if program is None:
        return

    def walk_block(block):
        for op in getattr(block, "operations", []) or []:
            yield op
            for nested in getattr(op, "blocks", []) or []:
                yield from walk_block(nested)

    for fn in getattr(program, "functions", {}).values():
        block = getattr(fn, "block", None)
        if block is not None:
            yield from walk_block(block)


def aggregate_label(counts: Counter) -> str:
    """Reduce a {device: op_count} histogram to one of ANE/GPU/CPU/MIXED."""
    if not counts:
        return "UNKNOWN"
    labels = [UNIT_LABEL.get(k, "?") for k in counts]
    nonzero = [(UNIT_LABEL.get(k, "?"), v) for k, v in counts.items() if v > 0]
    distinct = {l for l, _ in nonzero}
    if len(distinct) == 1:
        return next(iter(distinct))
    return "MIXED"


def resolve_one(pkg: Path, requested_unit: ct.ComputeUnit) -> tuple[str, dict]:
    """Run MLComputePlan and return (aggregate_label, per_device_op_count).

    MLComputePlan needs a compiled .mlmodelc, so we load the .mlpackage
    once via MLModel (which compiles it) and pass the compiled path.
    """
    model = ct.models.MLModel(str(pkg), compute_units=requested_unit)
    compiled_path = model.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(
        path=str(compiled_path),
        compute_units=requested_unit,
    )
    bucket = Counter()
    raw = Counter()
    op_details = []
    n_compute_ops = 0
    n_total_ops = 0
    for op in _walk_ops(plan.model_structure):
        n_total_ops += 1
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        # Constants and no-op metadata ops return None — only real compute ops
        # carry a device assignment.
        if usage is None:
            continue
        dev = getattr(usage, "preferred_compute_device", None)
        if dev is None:
            continue
        cls_name = type(dev).__name__
        if "Neural" in cls_name:
            key = "MLComputeDeviceTypeNeuralEngine"
        elif "GPU" in cls_name:
            key = "MLComputeDeviceTypeGPU"
        elif "CPU" in cls_name:
            key = "MLComputeDeviceTypeCPU"
        else:
            key = repr(dev)
        bucket[key] += 1
        raw[cls_name] += 1
        n_compute_ops += 1
        # Capture op output name for traceability
        out_name = None
        outs = getattr(op, "outputs", None) or []
        if outs:
            out_name = getattr(outs[0], "name", None)
        op_details.append({
            "output": out_name,
            "preferred": UNIT_LABEL.get(key, "?"),
            "supported": [UNIT_LABEL.get(_classify(d), "?") for d in
                          (getattr(usage, "supported_compute_devices", []) or [])],
        })
    return aggregate_label(bucket), {
        "n_total_ops": n_total_ops,
        "n_compute_ops": n_compute_ops,
        "by_device": dict(bucket),
        "by_class": dict(raw),
        "ops": op_details,
    }


def _classify(dev) -> str:
    cls_name = type(dev).__name__
    if "Neural" in cls_name:
        return "MLComputeDeviceTypeNeuralEngine"
    if "GPU" in cls_name:
        return "MLComputeDeviceTypeGPU"
    if "CPU" in cls_name:
        return "MLComputeDeviceTypeCPU"
    return cls_name


PKG_RE = re.compile(
    r"^router_(?P<shape>.+)_b(?P<batch>\d+)\.mlpackage$|"
    r"^expert_(?P<eshape>.+)_b(?P<ebatch>\d+)\.mlpackage$|"
    r"^block_(?P<blkkind>router|expert|all)_(?P<bshape>.+)_b(?P<bbatch>\d+)\.mlpackage$"
)


def parse_pkg_name(pkg_name: str) -> tuple[str, str, int] | None:
    """Map an .mlpackage filename back to (graph, shape, batch)."""
    m = PKG_RE.match(pkg_name)
    if not m:
        return None
    if m.group("shape"):
        return ("router_only", m.group("shape"), int(m.group("batch")))
    if m.group("eshape"):
        return ("expert", m.group("eshape"), int(m.group("ebatch")))
    if m.group("blkkind"):
        return (f"block_{m.group('blkkind')}", m.group("bshape"), int(m.group("bbatch")))
    return None


# Note: the router_coreml.py builder names a specific variant
# "router_router_stress_dense_*" because it concatenates a "router_" prefix
# onto the shape name "router_stress_dense". Strip the duplicate prefix
# so it matches the JSON's `shape` field.
def _normalize_shape(s: str) -> str:
    return s


REQUESTED_UNITS = [
    ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
    ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
    ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
    ("ALL", ct.ComputeUnit.ALL),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default=None, help="Filter to one shape name.")
    ap.add_argument("--graph", default=None,
                    help="Filter to one graph: router_only | expert | block_*.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print results without writing to result JSONs.")
    args = ap.parse_args()

    summary = []
    pkgs = sorted(MODELS_DIR.glob("*.mlpackage"))
    print(f"Found {len(pkgs)} mlpackage(s) in {MODELS_DIR}")

    # Build an index: (graph, shape, batch) -> resolved label per ComputeUnit.
    resolved: dict[tuple[str, str, int], dict[str, tuple[str, dict]]] = {}

    for pkg in pkgs:
        parsed = parse_pkg_name(pkg.name)
        if parsed is None:
            continue
        graph, shape, batch = parsed
        if args.shape and shape != args.shape:
            continue
        if args.graph and graph != args.graph:
            continue
        # Skip things we explicitly don't care about for actual_execution.
        if pkg.name.startswith("identity_") or pkg.name.startswith("pulse_"):
            continue
        per_unit = {}
        for unit_name, unit in REQUESTED_UNITS:
            try:
                label, detail = resolve_one(pkg, unit)
            except Exception as e:
                label, detail = "ERROR", {"error": str(e)[:300]}
            per_unit[unit_name] = (label, detail)
        resolved[(graph, shape, batch)] = per_unit
        labels_str = ", ".join(f"{u}:{l}" for u, (l, _) in per_unit.items())
        print(f"  {pkg.name:<60} -> {labels_str}")

    if args.dry_run:
        print("\n(--dry-run; not patching)")
        return

    # Patch matching JSON rows.
    patched = 0
    for path in RESULTS_DIR.glob("*.json"):
        if path.name == "null_experiments.json" or path.name == "energy_headline.json":
            continue
        try:
            with path.open() as f:
                row = json.load(f)
        except Exception:
            continue
        if "shape" not in row:
            continue
        key = (row["graph"], row["shape"], row["batch"])
        per_unit = resolved.get(key)
        if not per_unit:
            continue
        unit_name = row.get("compute_unit")
        if unit_name not in per_unit:
            continue
        label, detail = per_unit[unit_name]
        # Don't overwrite the rail-signature inferences for energy-confirmed
        # cells unless the compute-plan disagrees.
        prior = row.get("actual_execution")
        if prior in (None, "UNKNOWN", "ANE_INFERENCE_ERROR", "MIXED"):
            row["actual_execution"] = label
        elif prior != label:
            row["actual_execution"] = f"{prior}_BUT_PLAN_SAYS_{label}"
        row["compute_plan_op_counts"] = detail
        with path.open("w") as f:
            json.dump(row, f, indent=2, default=str)
        patched += 1

    print(f"\npatched {patched} result JSON(s).")


if __name__ == "__main__":
    main()
