"""Record what each machine can run: library versions, GPU, and Core ML device placement."""

from __future__ import annotations

import importlib
import platform
import statistics
import tempfile
import time

from faceid_bench.latency import machine_label

LIBRARIES = ("numpy", "cv2", "onnxruntime", "torch", "torchvision", "coremltools")


def library_versions(names: tuple[str, ...] = LIBRARIES) -> dict[str, str | None]:
    """Version of each library, or None when it is not installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def torch_cuda() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    x = torch.randn(1024, 1024, device="cuda")
    ok = bool(torch.isfinite(x @ x).all())
    return {
        "available": True,
        "device": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "vram_gib": round(props.total_memory / 2**30, 1),
        "matmul_ok": ok,
    }


def onnxruntime_providers() -> list[str]:
    try:
        import onnxruntime
    except ImportError:
        return []
    return onnxruntime.get_available_providers()


def coreml_timing(size: int = 640, runs: int = 100) -> dict[str, object]:
    """Time a small conv net on each Core ML compute unit.

    Core ML's compute plan reported CPU for every op on macOS 27 even when the Neural Engine
    ran the model 5.8x faster, so placement is judged by timing, not by the plan.
    """
    if platform.system() != "Darwin":
        return {"available": False}
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError:
        return {"available": False}

    rng = np.random.default_rng(0)
    weights, channels = [], 3
    for out in (32, 64, 128, 256, 256, 512):
        weights.append(rng.standard_normal((out, channels, 3, 3)).astype(np.float32))
        channels = out

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, size, size), dtype=types.fp32)])
    def net(x):
        for w in weights:
            x = mb.relu(x=mb.conv(x=x, weight=w, strides=[2, 2], pad_type="same"))
        return mb.reduce_mean(x=x, axes=[2, 3])

    x = {"x": rng.random((1, 3, size, size), dtype=np.float32)}
    p50_ms = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/probe.mlpackage"
        ct.convert(
            net,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT16,
        ).save(path)
        for unit in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"):
            model = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, unit))
            for _ in range(10):
                model.predict(x)
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                model.predict(x)
                times.append((time.perf_counter() - t0) * 1e3)
            p50_ms[unit] = round(statistics.median(times), 3)
    return {
        "available": True,
        "probe": f"6-conv net, {size}x{size}, fp16, batch 1, median of {runs}",
        "p50_ms": p50_ms,
        "ane_speedup_vs_cpu": round(p50_ms["CPU_ONLY"] / p50_ms["CPU_AND_NE"], 2),
    }


def collect() -> dict[str, object]:
    return {
        "machine": machine_label(),
        "system": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "libraries": library_versions(),
        "onnxruntime_providers": onnxruntime_providers(),
        "cuda": torch_cuda(),
        "coreml": coreml_timing(),
    }
