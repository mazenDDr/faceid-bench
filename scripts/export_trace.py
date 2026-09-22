"""One real unlock attempt, stage by stage, for the README animation. Runs on the Mac.

Person A enrolls from one LFW photo, then tries with another photo (genuine); person B tries
too (impostor). Uses the balanced pipeline on 640x480 frames, like the timing runs. Nothing
identifying is written: no names, no pixels, and the embedding only as 32 contributions to
the cosine (each summing 16 of 512 dimensions). Writes results/trace.json.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from faceid_bench.align import ARCFACE_112, umeyama
from faceid_bench.latency import providers_for
from faceid_bench.pipeline import Config, Pipeline

SAMPLE = Path("data/lfw_sample")
pipeline = json.loads(Path("results/pipeline.json").read_text())["configs"]["balanced"]
cfg = pipeline["config"]
pipe = Pipeline(
    Config(cfg["name"], cfg["detector"], cfg["size"], cfg["embedder"], tuple(cfg["platt"])),
    providers_for("coreml", "all"),
)


def frame(path: str) -> np.ndarray:
    face = cv2.resize(cv2.imread(str(SAMPLE / path)), (480, 480), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((480, 640, 3), 127, np.uint8)
    canvas[:, 80:560] = face
    return canvas


paths = SAMPLE.joinpath("index.txt").read_text().split()
enroll_path, genuine_path, impostor_path = paths[0], paths[1], paths[3]
enroll = pipe.embed(frame(enroll_path))


def attempt(path: str) -> dict:
    info: dict = {}
    emb = pipe.embed(frame(path), info=info)
    row = info["face"]
    landmarks = row[4:14].reshape(5, 2)
    matrix = umeyama(landmarks, ARCFACE_112)
    aligned = landmarks @ matrix[:, :2].T + matrix[:, 2]
    products = emb * enroll
    a, b = cfg["platt"]
    sim = float(products.sum())
    llr = a * sim + b
    return {
        "box": [round(float(v), 1) for v in row[:4]],
        "landmarks": np.round(landmarks, 1).tolist(),
        "aligned": np.round(aligned, 2).tolist(),
        "detector_score": round(float(row[14]), 3),
        "contributions": np.round(products.reshape(32, 16).sum(axis=1), 5).tolist(),
        "similarity": round(sim, 4),
        "llr": round(llr, 3),
        "probability": round(float(1 / (1 + np.exp(-llr))), 6),
    }


trace = {
    "frame": [640, 480],
    "template_112": ARCFACE_112.tolist(),
    "platt": cfg["platt"],
    "threshold": pipeline["operating_points"]["1e-05"]["threshold_from_dev"],
    "times_ms": pipeline["mac_coreml_all_fp16"],
    "genuine": attempt(genuine_path),
    "impostor": attempt(impostor_path),
}
Path("results/trace.json").write_text(json.dumps(trace, indent=1) + "\n")
print({k: (trace[k]["similarity"], trace[k]["probability"]) for k in ("genuine", "impostor")})
