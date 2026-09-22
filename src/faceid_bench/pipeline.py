"""The Face ID-like pipeline: find the face, align it, embed it, compare, calibrated probability.

One `verify` call is one unlock attempt against an enrolled template. Every stage is timed
separately so the total can be broken down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from faceid_bench.align import align
from faceid_bench.calibrate import probability
from faceid_bench.detectors import SCRFD
from faceid_bench.embedders import ArcFaceONNX


@dataclass(frozen=True)
class Config:
    name: str
    detector: str  # fixed-shape SCRFD file under models/
    size: int
    embedder: str  # batch-1 embedder file under models/
    platt: tuple[float, float]  # (a, b): LLR = a * cosine + b, fit on dev people


class Pipeline:
    def __init__(self, config: Config, providers: list):
        self.config = config
        self.detector = SCRFD(config.detector, size=config.size, providers=providers)
        self.embedder = ArcFaceONNX(config.embedder, providers=providers)

    @staticmethod
    def pick(det: np.ndarray) -> np.ndarray | None:
        """The largest face with score >= 0.5: the person holding the phone."""
        det = det[det[:, 14] >= 0.5]
        if len(det) == 0:
            return None
        return det[np.argmax(det[:, 2] * det[:, 3])]

    def embed(self, frame: np.ndarray, times: dict | None = None) -> np.ndarray | None:
        times = {} if times is None else times
        t0 = time.perf_counter()
        face = self.pick(self.detector(frame))
        t1 = time.perf_counter()
        times["detect_ms"] = (t1 - t0) * 1e3
        if face is None:
            return None
        crop = align(frame, face[4:14])
        t2 = time.perf_counter()
        emb = self.embedder.embed_crops(crop[None])[0]
        t3 = time.perf_counter()
        times["align_ms"], times["embed_ms"] = (t2 - t1) * 1e3, (t3 - t2) * 1e3
        return emb

    def verify(self, frame: np.ndarray, template: np.ndarray, prior: float = 0.5) -> dict:
        """One unlock attempt: returns probability, LLR, similarity and per-stage times."""
        start, times = time.perf_counter(), {}
        emb = self.embed(frame, times)
        t = time.perf_counter()
        if emb is None:
            result = {"face": False, "probability": 0.0}
        else:
            a, b = self.config.platt
            sim = float(emb @ template)
            llr = a * sim + b
            result = {
                "face": True,
                "similarity": sim,
                "llr": llr,
                "probability": float(probability(llr, prior)),
            }
        times["decide_ms"] = (time.perf_counter() - t) * 1e3
        times["total_ms"] = (time.perf_counter() - start) * 1e3
        return {**result, "times": times}
