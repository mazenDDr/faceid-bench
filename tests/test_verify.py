import numpy as np
import pytest

from faceid_bench.align import ARCFACE_112, umeyama
from faceid_bench.verify import (
    PairScores,
    bootstrap_rates,
    lfw_accuracy,
    threshold_at_far,
)


def test_umeyama_recovers_a_similarity_transform():
    angle, scale, shift = 0.3, 1.7, np.array([12.0, -4.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = ARCFACE_112 + 5
    dst = scale * src @ rot.T + shift
    m = umeyama(src, dst)
    np.testing.assert_allclose(m[:, :2], scale * rot, atol=1e-9)
    np.testing.assert_allclose(m[:, 2], shift, atol=1e-9)


def test_lfw_accuracy_perfect_when_separable():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(600, 8))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    same = np.arange(600) % 2 == 0
    b = np.where(same[:, None], a, -a)  # same: identical, different: opposite
    acc, std = lfw_accuracy(a, b, same, np.repeat(np.arange(10), 60))
    assert acc == 1.0 and std == 0.0


def test_threshold_at_far_accepts_at_most_target():
    scores = np.arange(10000) / 10000
    t = threshold_at_far(scores, 1e-3)
    assert (scores > t).mean() <= 1e-3
    assert (scores > t - 1e-4).mean() > 0  # not needlessly strict


def toy_scores():
    rng = np.random.default_rng(1)
    identity = np.repeat(np.arange(30), rng.integers(1, 5, 30))
    centers = rng.normal(size=(30, 16))
    emb = centers[identity] + 0.6 * rng.normal(size=(len(identity), 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return PairScores(emb, identity)


def test_pair_counts_match_all_vs_all():
    s = toy_scores()
    n = len(s.identity)
    assert len(s.genuine) + len(s.impostor) == n * (n - 1) // 2
    assert len(s.genuine) == int((s.sizes * (s.sizes - 1) // 2).sum())


@pytest.mark.parametrize("t", [0.0, 0.3, 0.6])
def test_bootstrap_with_unit_weights_equals_point_rates(t):
    s = toy_scores()
    counts = s.per_identity(t)
    tar, far = bootstrap_rates(counts, s.sizes.astype(float), np.ones((2, s.n_ids)))
    point = s.rates(t)
    assert tar == pytest.approx([point["tar"]] * 2)
    assert far == pytest.approx([point["far"]] * 2)
