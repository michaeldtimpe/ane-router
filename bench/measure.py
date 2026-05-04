"""Timing harness, power sampler, null calibration helpers, result writer.

Design notes:
- Latency uses time.perf_counter_ns. N=1000 measured iters, 100 warmup default.
- First call is timed separately (cold dispatch / Core ML compile penalty).
- PowerSampler shells out to sudo powermetrics. Caller pre-authenticates with
  `sudo -v`. The sampler records all samples while the context is active;
  caller is responsible for keeping only steady-state work inside the context.
- Idle subtraction: idle_baseline_mw() runs PowerSampler with no workload for
  N seconds, returns the per-rail floor. Steady-state mW reported with every
  result is the delta over this floor.
- measure_null_roundtrip and measure_coreml_empty produce boundary_us and
  dispatch_us — the floor for any hybrid configuration.
"""
from __future__ import annotations

import json
import os
import platform
import plistlib
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "raw"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class HostInfo:
    machine: str
    macos: str
    thermal_pressure: str
    ac_power: bool

    @classmethod
    def collect(cls) -> "HostInfo":
        try:
            therm = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True, text=True, timeout=2,
            ).stdout
        except Exception:
            therm = ""
        thermal_pressure = "nominal"
        for line in therm.splitlines():
            if "CPU_Speed_Limit" in line:
                # If CPU speed is limited <100, we're throttled
                try:
                    pct = int(line.split("=")[1].strip())
                    if pct < 100:
                        thermal_pressure = f"throttled_{pct}"
                except Exception:
                    pass
        try:
            psstat = subprocess.run(
                ["pmset", "-g", "ps"],
                capture_output=True, text=True, timeout=2,
            ).stdout
            ac = "AC Power" in psstat
        except Exception:
            ac = False
        return cls(
            machine=platform.machine(),
            macos=platform.mac_ver()[0],
            thermal_pressure=thermal_pressure,
            ac_power=ac,
        )


@dataclass
class TimingResult:
    median_ms: float
    p95_ms: float
    p99_ms: float
    stddev_ms: float
    first_call_ms: float
    n_iters: int
    n_warmup: int

    @classmethod
    def from_ns(cls, latencies_ns: list[int], first_call_ns: int, n_warmup: int) -> "TimingResult":
        arr = np.asarray(latencies_ns, dtype=np.float64) / 1e6
        return cls(
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            stddev_ms=float(arr.std()),
            first_call_ms=first_call_ns / 1e6,
            n_iters=len(arr),
            n_warmup=n_warmup,
        )


@dataclass
class ResultRow:
    shape: str
    graph: str  # router_only | router_topk | expert | block | null
    backend: str
    compute_unit: str | None
    actual_execution: str  # ANE | GPU | CPU | MIXED | ANE_FALLBACK | UNKNOWN
    batch: int
    precision: str
    timing: TimingResult
    boundary_us: float = 0.0
    dispatch_us: float = 0.0
    kernel_us: float = 0.0
    throughput_tok_s: float = 0.0
    mj_per_token: float = 0.0
    mw_idle_baseline: dict = field(default_factory=dict)
    mw_steady: dict = field(default_factory=dict)
    verify_ane_passed: bool = False
    instruments_trace: str | None = None
    coreml_warnings_log: str | None = None
    host: dict = field(default_factory=dict)
    ts: str = ""
    notes: str = ""

    def write(self, name: str | None = None) -> Path:
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat()
        if not self.host:
            self.host = asdict(HostInfo.collect())
        # Include compute_unit and ns timestamp so multiple writes within the
        # same second don't collide.
        cu = (self.compute_unit or "x").lower()
        suffix = f"{time.monotonic_ns():x}"
        fname = name or (
            f"{self.shape}_{self.graph}_{self.backend}_{cu}_b{self.batch}_{suffix}.json"
        )
        path = RESULTS_DIR / fname
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


def benchmark(
    func: Callable[[], Any],
    iters: int = 1000,
    warmup: int = 100,
    max_seconds: float = 5.0,
    min_iters: int = 30,
    max_warmup_seconds: float = 1.5,
) -> TimingResult:
    """Run func() repeatedly, return latency statistics.

    First call is timed separately (cold dispatch).
    Caller is responsible for ensuring func() forces full evaluation
    (e.g. mlx.eval()).

    Both warmup and measurement are time-capped so a single heavy cell
    can't run for hours: warmup stops after `max_warmup_seconds` (still
    runs >=1 iter), measurement stops after `max_seconds` (still runs
    `min_iters`).
    """
    t0 = time.perf_counter_ns()
    func()
    first_ns = time.perf_counter_ns() - t0

    warmup_deadline = time.perf_counter_ns() + int(max_warmup_seconds * 1e9)
    for _ in range(max(0, warmup - 1)):
        if time.perf_counter_ns() > warmup_deadline:
            break
        func()

    latencies: list[int] = []
    deadline = time.perf_counter_ns() + int(max_seconds * 1e9)
    for i in range(iters):
        t = time.perf_counter_ns()
        func()
        latencies.append(time.perf_counter_ns() - t)
        if i + 1 >= min_iters and time.perf_counter_ns() > deadline:
            break

    return TimingResult.from_ns(latencies, first_ns, warmup)


# ---------------------------------------------------------------------------
# powermetrics
# ---------------------------------------------------------------------------


class PowerSampler:
    """Background sudo powermetrics sampler.

    Caller pre-authenticates with `sudo -v`. The sampler uses `sudo -n` so it
    will fail fast with a clear error if the timestamp expired.

    Usage:
        baseline = idle_baseline_mw(seconds=10)  # before any workload
        ps = PowerSampler()
        ps.start()
        # ... warmup ...
        ps.mark_steady_start()
        # ... 30 s workload ...
        ps.mark_steady_stop()
        ps.stop()
        steady = ps.steady_state_mw()
        delta = {k: max(0.0, steady.get(k, 0.0) - baseline.get(k, 0.0)) for k in steady}
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen | None = None
        self.samples: list[dict] = []
        self._t_start: float = 0.0
        self._steady_start: float | None = None
        self._steady_stop: float | None = None
        self._tmp_out: str | None = None

    def start(self) -> None:
        # Write to a temp file rather than pipe stdout: a single plist sample
        # is ~28 KB, exceeding the default pipe buffer; after a couple of
        # samples powermetrics blocks if we don't drain. With -o, the kernel
        # handles flushing.
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".plist", prefix="pm_")
        os.close(fd)
        self._tmp_out = path
        cmd = [
            "sudo", "-n", "powermetrics",
            "--samplers", "cpu_power,gpu_power,ane_power",
            "-i", str(self.interval_ms),
            "-o", path,
            "--format", "plist",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        self._t_start = time.perf_counter()

    def mark_steady_start(self) -> None:
        self._steady_start = time.perf_counter()

    def mark_steady_stop(self) -> None:
        self._steady_stop = time.perf_counter()

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), 15)
        except ProcessLookupError:
            pass
        try:
            _, stderr = self.proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            _, stderr = self.proc.communicate()
        if self._tmp_out and Path(self._tmp_out).exists():
            with open(self._tmp_out, "rb") as f:
                raw = f.read()
            try:
                os.unlink(self._tmp_out)
            except OSError:
                pass
            if not raw and stderr:
                raise RuntimeError(
                    f"powermetrics produced no data; stderr: {stderr[:500]!r}"
                )
            self._parse(raw)
        elif stderr:
            raise RuntimeError(
                f"powermetrics produced no data; stderr: {stderr[:500]!r}"
            )

    def __enter__(self) -> "PowerSampler":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def _parse(self, raw: bytes) -> None:
        # powermetrics emits concatenated plist docs separated by NUL bytes.
        # Each doc carries `elapsed_ns` (since the previous sample) — the
        # `timestamp` field has 1-second resolution so it's useless for
        # sub-second windows. We accumulate elapsed_ns to build cumulative
        # time, with the sampler entry-time as origin.
        chunks = raw.split(b"\x00")
        cum_ns = 0
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk.startswith(b"<?xml"):
                continue
            try:
                doc = plistlib.loads(chunk)
            except Exception:
                continue
            elapsed_ns = doc.get("elapsed_ns")
            if isinstance(elapsed_ns, int):
                cum_ns += elapsed_ns
            cpu_mw = _walk_for_first_number(doc, ["cpu_power", "package_power"])
            gpu_mw = _walk_for_first_number(doc, ["gpu_power"])
            ane_mw = _walk_for_first_number(doc, ["ane_power"])
            self.samples.append(
                {
                    # Cumulative elapsed seconds from sampler start. The first
                    # sample's elapsed_ns is the time from process spawn to
                    # first sample, so we start at that point as t=elapsed_s.
                    "elapsed_s": cum_ns / 1e9,
                    "cpu_mw": cpu_mw,
                    "gpu_mw": gpu_mw,
                    "ane_mw": ane_mw,
                }
            )

    def _window_samples(self) -> list[dict]:
        if self._steady_start is None or self._steady_stop is None:
            return list(self.samples)
        rel_start = self._steady_start - self._t_start
        rel_stop = self._steady_stop - self._t_start
        out = []
        for s in self.samples:
            e = s.get("elapsed_s")
            if e is None:
                continue
            if rel_start <= e <= rel_stop:
                out.append(s)
        return out

    def steady_state_mw(self) -> dict:
        win = self._window_samples()
        if not win:
            return {"cpu_mw": 0.0, "gpu_mw": 0.0, "ane_mw": 0.0, "n_samples": 0}
        return {
            "cpu_mw": _mean([s["cpu_mw"] for s in win]),
            "gpu_mw": _mean([s["gpu_mw"] for s in win]),
            "ane_mw": _mean([s["ane_mw"] for s in win]),
            "n_samples": len(win),
        }


def _mean(xs: list[float | None]) -> float:
    vals = [x for x in xs if x is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _walk_for_first_number(d: Any, keys: list[str]) -> float | None:
    """Find the first numeric value at any nested key in `keys`. None if absent."""
    stack: list[Any] = [d]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            for k, vv in v.items():
                if k in keys and isinstance(vv, (int, float)):
                    return float(vv)
                if isinstance(vv, (dict, list)):
                    stack.append(vv)
        elif isinstance(v, list):
            stack.extend(v)
    return None


def idle_baseline_mw(seconds: float = 10.0, interval_ms: int = 100) -> dict:
    """Sample idle power for N seconds with no workload.

    Returns dict with cpu_mw / gpu_mw / ane_mw averages.
    """
    ps = PowerSampler(interval_ms=interval_ms)
    ps.start()
    ps.mark_steady_start()
    time.sleep(seconds)
    ps.mark_steady_stop()
    ps.stop()
    return ps.steady_state_mw()


# ---------------------------------------------------------------------------
# Null calibration: pure boundary cost, pure CoreML dispatch overhead
# ---------------------------------------------------------------------------


def measure_null_roundtrip(batch: int, hidden: int = 4096, iters: int = 1000) -> TimingResult:
    """MLX tensor -> numpy -> MLX tensor. No CoreML in the loop.

    Isolates the cost of the framework bridge that any hybrid configuration
    must pay per crossing.
    """
    import mlx.core as mx

    src = mx.array(np.random.randn(batch, hidden).astype(np.float16))
    mx.eval(src)

    def step():
        npy = np.asarray(src, dtype=np.float16)
        out = mx.array(npy)
        mx.eval(out)

    return benchmark(step, iters=iters, warmup=100)


def measure_coreml_empty(batch: int, hidden: int = 4096, iters: int = 1000) -> tuple[TimingResult, str]:
    """MLX tensor -> CoreML identity model -> MLX tensor.

    Subtract measure_null_roundtrip from this to get pure Core ML
    dispatch overhead.

    Returns (timing, actual_execution_label).
    """
    import coremltools as ct
    import mlx.core as mx
    import torch
    import torch.nn as nn

    cache_path = MODELS_DIR / f"identity_b{batch}_h{hidden}.mlpackage"
    if not cache_path.exists():
        class Identity(nn.Module):
            def forward(self, x):
                return x

        ex = torch.randn(batch, hidden, 1, 1, dtype=torch.float16)
        traced = torch.jit.trace(Identity().eval().half(), ex)
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=ex.shape, name="x", dtype=np.float16)],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
        )
        ml.save(str(cache_path))

    model = ct.models.MLModel(str(cache_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    src = mx.array(np.random.randn(batch, hidden).astype(np.float16))
    mx.eval(src)

    def step():
        npy = np.asarray(src, dtype=np.float16).reshape(batch, hidden, 1, 1)
        out = model.predict({"x": npy})["y"]
        back = mx.array(out.reshape(batch, hidden))
        mx.eval(back)

    timing = benchmark(step, iters=iters, warmup=100)
    return timing, "UNKNOWN"  # actual_execution to be filled in by Instruments inspection


# ---------------------------------------------------------------------------
# CoreML conversion-warning capture
# ---------------------------------------------------------------------------


@contextmanager
def capture_coreml_warnings(out_path: Path):
    """Capture stderr from coremltools.convert into a file.

    Usage:
        with capture_coreml_warnings(Path("results/raw/run.warnings.log")):
            ml = ct.convert(...)
    """
    import io
    import logging
    import sys

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("coremltools")
    handler = logging.FileHandler(out_path, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)

    saved_stderr = sys.stderr
    buf = io.StringIO()

    class Tee:
        def write(self, s):
            buf.write(s)
            saved_stderr.write(s)
        def flush(self):
            saved_stderr.flush()

    sys.stderr = Tee()
    try:
        yield
    finally:
        sys.stderr = saved_stderr
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        handler.close()
        # Also append captured stderr text after the structured log
        with out_path.open("a") as f:
            f.write("\n--- captured stderr ---\n")
            f.write(buf.getvalue())
