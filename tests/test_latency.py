import itertools

import numpy as np
import pytest

from faceid_bench.latency import measure, slug, summarize


def test_measure_runs_warmup_untimed_then_times_each_run():
    calls = []
    ticks = itertools.count(step=0.002)  # every clock read advances 2 ms
    times = measure(lambda: calls.append(1), warmup=3, runs=5, clock=lambda: next(ticks))
    assert len(calls) == 8
    assert times == pytest.approx([2.0] * 5)


def test_summarize_percentiles():
    s = summarize([float(i) for i in range(1, 101)])
    assert s["p50_ms"] == pytest.approx(50.5)
    assert s["p95_ms"] == pytest.approx(95.05)
    assert s["min_ms"] == 1.0


def test_slug_makes_hostname_free_labels():
    assert slug(" GeForce RTX 5060 Ti") == "geforcertx5060ti"
    assert "mac-" + slug(" M4 Pro") == "mac-m4pro"


def tiny_onnx(path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.ones((8, 3, 3, 3), np.float32), "w")
    node = helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 32, 32])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)


def test_time_onnx_cpu_records_protocol(tmp_path):
    pytest.importorskip("onnxruntime")
    from faceid_bench.latency import time_onnx

    path = str(tmp_path / "tiny.onnx")
    tiny_onnx(path)
    t = time_onnx(path, "cpu", warmup=2, runs=5)
    assert t.input_shape == [1, 3, 32, 32]
    assert t.runs == 5 and t.p50_ms > 0 and t.p95_ms >= t.p50_ms
    assert t.active == ["CPUExecutionProvider"] and t.note == ""


def test_missing_provider_is_reported_not_hidden(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from faceid_bench.latency import time_onnx

    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("CUDA present; fallback path not reachable here")
    path = str(tmp_path / "tiny.onnx")
    tiny_onnx(path)
    t = time_onnx(path, "cuda", warmup=1, runs=2)
    assert t.active[0] == "CPUExecutionProvider"
    assert "not active" in t.note


def test_median_session_picks_middle_p50():
    from faceid_bench.latency import median_session

    sessions = [
        {"p50_ms": 6.7, "p95_ms": 7.0},
        {"p50_ms": 2.6, "p95_ms": 3.0},
        {"p50_ms": 2.7, "p95_ms": 3.1},
    ]
    assert median_session(sessions) == {"p50_ms": 2.7, "p95_ms": 3.1}
    assert median_session(sessions[:2])["p50_ms"] == 2.6


def test_time_onnx_keeps_every_session(tmp_path):
    pytest.importorskip("onnxruntime")
    from faceid_bench.latency import time_onnx

    path = str(tmp_path / "tiny.onnx")
    tiny_onnx(path)
    t = time_onnx(path, "cpu", warmup=1, runs=3, sessions=3)
    assert t.sessions == 3 and len(t.session_p50_ms) == 3
    assert t.p50_ms == sorted(t.session_p50_ms)[1]
