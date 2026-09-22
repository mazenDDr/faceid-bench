"""One timing protocol for every backend: warm-up, batch 1, then p50/p95 over many runs.

A latency number is only reported together with the machine label, backend, compute unit,
input shape and run counts, so every record carries all of them.
"""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Friendly names for onnxruntime providers and Core ML compute units.
ORT_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}
COREML_UNITS = {
    "cpu": ("CPU_ONLY", "CPUOnly"),
    "gpu": ("CPU_AND_GPU", "CPUAndGPU"),
    "ane": ("CPU_AND_NE", "CPUAndNeuralEngine"),
    "all": ("ALL", "ALL"),
}


@dataclass
class Timing:
    model: str
    backend: str
    unit: str
    machine: str
    input_shape: list[int]
    warmup: int
    runs: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    load_ms: float
    active: list[str] = field(default_factory=list)
    note: str = ""


def machine_label() -> str:
    """Short hardware label such as `mac-m4pro` or `rtx5060ti`. Never the hostname."""
    if label := os.environ.get("FACEID_MACHINE"):
        return label
    try:
        import torch

        if torch.cuda.is_available():
            return slug(torch.cuda.get_device_name(0).replace("NVIDIA", "").replace("GeForce", ""))
    except ImportError:
        pass
    if platform.system() == "Darwin":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout
        return "mac-" + slug(chip.replace("Apple", ""))
    return slug(platform.machine())


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def summarize(times_ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(times_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 3),
        "mean_ms": round(statistics.fmean(times_ms), 3),
        "min_ms": round(min(times_ms), 3),
    }


def measure(
    run: Callable[[], object],
    warmup: int = 20,
    runs: int = 200,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Call `run` warmup times untimed, then `runs` times timed. Returns milliseconds.

    `run` must block until the result is ready (outputs copied back to the host), so GPU
    work is inside the timed region.
    """
    for _ in range(warmup):
        run()
    times = []
    for _ in range(runs):
        start = clock()
        run()
        times.append((clock() - start) * 1e3)
    return times


def onnx_session(path: str, backend: str, unit: str = "all"):
    """onnxruntime session on one provider, CPU fallback allowed but reported via `active`."""
    import onnxruntime as ort

    provider = ORT_PROVIDERS[backend]
    if backend == "coreml":
        options = {"ModelFormat": "MLProgram", "MLComputeUnits": COREML_UNITS[unit][1]}
        providers = [(provider, options), "CPUExecutionProvider"]
    elif backend == "cpu":
        providers = [provider]
    else:
        providers = [provider, "CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    return ort.InferenceSession(path, session_options, providers=providers)


def time_onnx(
    path: str, backend: str, unit: str = "all", warmup: int = 20, runs: int = 200
) -> Timing:
    start = time.perf_counter()
    session = onnx_session(path, backend, unit)
    load_ms = (time.perf_counter() - start) * 1e3
    feed = random_feed(session)
    shape = list(next(iter(feed.values())).shape)
    times = measure(lambda: session.run(None, feed), warmup, runs)
    active = session.get_providers()
    wanted = ORT_PROVIDERS[backend]
    return Timing(
        model=Path(path).stem,
        backend=f"ort-{backend}",
        unit=unit if backend == "coreml" else backend,
        machine=machine_label(),
        input_shape=shape,
        warmup=warmup,
        runs=runs,
        load_ms=round(load_ms, 1),
        active=active,
        note="" if active[0] == wanted else f"{wanted} not active; ran on {active[0]}",
        **summarize(times),
    )


def time_coreml(path: str, unit: str = "ane", warmup: int = 20, runs: int = 200) -> Timing:
    """Native Core ML (.mlpackage). Placement is judged by comparing units, not the plan."""
    import coremltools as ct

    start = time.perf_counter()
    model = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, COREML_UNITS[unit][0]))
    load_ms = (time.perf_counter() - start) * 1e3
    spec_input = model.get_spec().description.input[0]
    shape = list(spec_input.type.multiArrayType.shape)
    x = {spec_input.name: np.random.default_rng(0).random(shape, dtype=np.float32)}
    times = measure(lambda: model.predict(x), warmup, runs)
    return Timing(
        model=Path(path).stem,
        backend="coreml",
        unit=unit,
        machine=machine_label(),
        input_shape=shape,
        warmup=warmup,
        runs=runs,
        load_ms=round(load_ms, 1),
        **summarize(times),
    )


def random_feed(session) -> dict[str, np.ndarray]:
    """Random batch-1 inputs; symbolic dims other than batch must be fixed in the model."""
    rng = np.random.default_rng(0)
    feed = {}
    for spec in session.get_inputs():
        shape = [1 if not isinstance(d, int) else d for d in spec.shape]
        feed[spec.name] = rng.random(shape, dtype=np.float32)
    return feed


def append(result: Timing, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
