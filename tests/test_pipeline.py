import numpy as np
import pytest

from faceid_bench.align import ARCFACE_112
from faceid_bench.pipeline import Config, Pipeline


def det_row(x, y, w, h, score):
    row = np.zeros(15, np.float32)
    row[:4], row[14] = (x, y, w, h), score
    row[4:14] = (ARCFACE_112 * w / 112 + (x, y)).reshape(-1)
    return row


class FakeDetector:
    def __init__(self, rows):
        self.rows = np.array(rows, np.float32).reshape(-1, 15)

    def __call__(self, frame):
        return self.rows.copy()


class FakeEmbedder:
    def embed_crops(self, crops):
        v = np.ones((len(crops), 4))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def pipeline(rows, platt=(10.0, -5.0)):
    p = Pipeline.__new__(Pipeline)
    p.config = Config("t", "", 320, "", platt)
    p.detector, p.embedder = FakeDetector(rows), FakeEmbedder()
    return p


def test_pick_takes_largest_confident_face():
    small, big, unsure = (
        det_row(0, 0, 40, 40, 0.9),
        det_row(50, 50, 120, 120, 0.8),
        det_row(0, 0, 300, 300, 0.3),
    )
    assert Pipeline.pick(np.stack([small, big, unsure]))[2] == 120
    assert Pipeline.pick(np.stack([unsure])) is None


def test_verify_uses_platt_and_prior():
    frame = np.zeros((480, 640, 3), np.uint8)
    template = np.ones(4) / 2  # cosine 1.0 with the fake embedding
    r = pipeline([det_row(100, 100, 200, 200, 0.95)]).verify(frame, template)
    assert r["face"] and r["similarity"] == pytest.approx(1.0)
    assert r["llr"] == pytest.approx(5.0)
    assert r["probability"] == pytest.approx(1 / (1 + np.exp(-5.0)))
    assert set(r["times"]) == {"detect_ms", "align_ms", "embed_ms", "decide_ms", "total_ms"}


def test_no_face_means_no_unlock():
    r = pipeline([]).verify(np.zeros((480, 640, 3), np.uint8), np.ones(4) / 2)
    assert not r["face"] and r["probability"] == 0.0
