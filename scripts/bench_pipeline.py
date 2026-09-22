"""Time the whole pipeline on the Mac as unlock attempts, and check Core ML against the CPU.

Frames are 640x480 with the face filling most of it (an LFW photo scaled to 480x480 on a grey
background), like a phone held at arm's length. Each of 100 test people is enrolled from one
photo; attempts use their second photo (genuine) and the next person's (impostor).
Writes outputs/pipeline/bench_<machine>.json.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from faceid_bench.latency import machine_label, providers_for
from faceid_bench.pipeline import Config, Pipeline

SAMPLE = Path("data/lfw_sample")
CAL = json.loads(Path("outputs/calibration/compare_pipeline.json").read_text())["models"]


def platt(model: str) -> tuple[float, float]:
    p = CAL[model]["calibrators"]["platt"]["params"]
    return p["a"], p["b"]


CONFIGS = [
    Config(
        "fast",
        "fixed/scrfd_500m_kps_320_fp16.onnx",
        320,
        "fixed/mbf_w600k_112_fp16.onnx",
        platt("mbf_w600k__scrfd_500m_kps@320"),
    ),
    Config(
        "balanced",
        "fixed/scrfd_10g_kps_320_fp16.onnx",
        320,
        "fixed/r50_w600k_112_fp16.onnx",
        platt("r50_w600k__scrfd_10g_kps@320"),
    ),
    Config(
        "accurate",
        "fixed/scrfd_10g_kps_640_fp16.onnx",
        640,
        "fixed/r50_w600k_112_fp16.onnx",
        platt("r50_w600k"),
    ),
]


def fp32(config: Config) -> Config:
    """Same pipeline with the fp32 models, as the CPU reference."""
    return Config(
        config.name,
        config.detector.replace("_fp16", ""),
        config.size,
        config.embedder.replace("_fp16", ""),
        config.platt,
    )


def frame(path: Path) -> np.ndarray:
    face = cv2.resize(cv2.imread(str(path)), (480, 480), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((480, 640, 3), 127, np.uint8)
    canvas[:, 80:560] = face
    return canvas


parser = argparse.ArgumentParser()
parser.add_argument("--sessions", type=int, default=3)
parser.add_argument("--laya", action="store_true", help="also time balanced + Laya decision")
args = parser.parse_args()

paths = SAMPLE.joinpath("index.txt").read_text().split()
frames = [frame(SAMPLE / p) for p in paths]
enroll, probe = frames[0::2], frames[1::2]
attempts = [(i, probe[i], True) for i in range(len(probe))]
attempts += [(i, probe[(i + 1) % len(probe)], False) for i in range(len(probe))]


def run(pipe: Pipeline, decide=None):
    templates = [pipe.embed(f) for f in enroll]
    for i, f, _ in attempts[:20]:  # warm-up
        pipe.verify(f, templates[i])
    rows = []
    for i, f, same in attempts:
        t0 = time.perf_counter()
        r = pipe.verify(f, templates[i])
        if decide is not None and r["face"]:
            t = time.perf_counter()
            r["probability"] = decide(r["similarity"])
            r["times"]["decide_ms"] = (time.perf_counter() - t) * 1e3
        r["times"]["total_ms"] = (time.perf_counter() - t0) * 1e3
        rows.append({**r, "same": same})
    return templates, rows


def summary(rows):
    stages = rows[0]["times"].keys()
    out = {s: round(statistics.median(r["times"].get(s, 0.0) for r in rows), 3) for s in stages}
    out["total_p95_ms"] = round(float(np.percentile([r["times"]["total_ms"] for r in rows], 95)), 3)
    gen = [r["probability"] for r in rows if r["same"]]
    imp = [r["probability"] for r in rows if not r["same"]]
    out["genuine_accepted_at_p0.5"] = sum(p > 0.5 for p in gen)
    out["impostor_accepted_at_p0.5"] = sum(p > 0.5 for p in imp)
    out["no_face"] = sum(not r["face"] for r in rows)
    return out


result = {"machine": machine_label(), "frames": "640x480, face 480x480", "attempts": len(attempts)}
for config in CONFIGS:
    sessions = []
    for _ in range(args.sessions):
        pipe = Pipeline(config, providers_for("coreml", "all"))
        templates, rows = run(pipe)
        sessions.append(summary(rows))
    median = sorted(sessions, key=lambda s: s["total_ms"])[(len(sessions) - 1) // 2]

    # fidelity: the same frames through the fp32 models on the CPU
    ref = Pipeline(fp32(config), providers_for("cpu"))
    ml = [pipe.embed(f) for f in probe]
    cpu = [ref.embed(f) for f in probe]
    cos = [float(a @ b) for a, b in zip(ml, cpu, strict=True) if a is not None and b is not None]
    lm = []
    for f in probe[:50]:
        a, b = pipe.pick(pipe.detector(f)), ref.pick(ref.detector(f))
        if a is not None and b is not None:
            lm.append(float(np.abs(a[4:14] - b[4:14]).max()))
    result[config.name] = {
        "config": config.__dict__,
        "coreml_all_fp16": median,
        "session_total_p50_ms": [s["total_ms"] for s in sessions],
        "fidelity_vs_cpu_fp32": {
            "embedding_cosine_min": round(min(cos), 5),
            "embedding_cosine_mean": round(float(np.mean(cos)), 6),
            "landmark_max_abs_px": round(max(lm), 3),
            "frames": len(cos),
        },
    }
    print(config.name, median, result[config.name]["fidelity_vs_cpu_fp32"], flush=True)

if args.laya:
    from faceid_bench.laya_decision import QUESTION, LayaJudge, state

    judge = LayaJudge(device="mps")
    no_det = np.zeros(15)

    def laya_decide(sim):
        answer = judge.agent.predict(state("note", sim, no_det, no_det), QUESTION)
        return answer["answers"]["same_person"]["noul"]

    config = CONFIGS[1]
    sessions = []
    for _ in range(args.sessions):
        _, rows = run(Pipeline(config, providers_for("coreml", "all")), decide=laya_decide)
        sessions.append(summary(rows))
    median = sorted(sessions, key=lambda s: s["total_ms"])[(len(sessions) - 1) // 2]
    result["balanced+laya"] = {
        "config": {**config.__dict__, "decision": "laya note variant, Mac GPU (MPS)"},
        "coreml_all_fp16": median,
        "session_total_p50_ms": [s["total_ms"] for s in sessions],
    }
    print("balanced+laya", median, flush=True)

out = Path(f"outputs/pipeline/bench_{result['machine']}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(f"saved {out}")
