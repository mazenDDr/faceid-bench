"""Turn a similarity score into a calibrated log-likelihood ratio (LLR).

The LLR says how much more likely the score is for the same person than for two different
people. It does not depend on how often genuine attempts happen; the probability does:
    P(same | score) = sigmoid(LLR + log(prior / (1 - prior))).
Calibrators are fit with both classes weighted equally (prior 0.5), so their output is an LLR.
Quality is scored with Cllr, a strictly proper scoring rule over all priors (Brümmer & du Preez,
2006): 0 is perfect, 1 is a system that always says "no idea".
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    """Weights that give each class half the total mass."""
    labels = np.asarray(labels, bool)
    w = np.empty(len(labels))
    w[labels] = 0.5 / labels.sum()
    w[~labels] = 0.5 / (~labels).sum()
    return w


class Naive:
    """The common shortcut: read (cos + 1) / 2 as a probability. Not fitted."""

    name = "naive"

    def fit(self, scores, labels):
        return self

    def llr(self, scores):
        p = np.clip((np.asarray(scores) + 1) / 2, EPS, 1 - EPS)
        return np.log(p / (1 - p))


class Platt:
    """LLR = a * score + b, fit by weighted logistic regression (Newton's method)."""

    name = "platt"

    def fit(self, scores, labels, iterations: int = 50):
        x = np.column_stack([np.asarray(scores, float), np.ones(len(scores))])
        y = np.asarray(labels, float)
        w = balanced_weights(labels)
        theta = np.zeros(2)
        for _ in range(iterations):
            p = 1 / (1 + np.exp(-x @ theta))
            grad = x.T @ (w * (p - y))
            hess = (x * (w * p * (1 - p))[:, None]).T @ x
            step = np.linalg.solve(hess + 1e-12 * np.eye(2), grad)
            theta -= step
            if np.abs(step).max() < 1e-10:
                break
        self.a, self.b = theta
        return self

    def llr(self, scores):
        return self.a * np.asarray(scores) + self.b


def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the non-decreasing fit to y (already sorted by score)."""
    values, weights, counts = [], [], []
    for yi, wi in zip(y, w, strict=True):
        values.append(yi)
        weights.append(wi)
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            wsum = weights[-2] + weights[-1]
            values[-2] = (values[-2] * weights[-2] + values[-1] * weights[-1]) / wsum
            weights[-2] = wsum
            counts[-2] += counts[-1]
            del values[-1], weights[-1], counts[-1]
    return np.repeat(values, counts)


class Isotonic:
    """Monotone step function from score to balanced probability, read as an LLR.

    Between fitted scores the probability is interpolated; it is clipped away from 0 and 1
    because a finite dev set cannot justify infinite evidence.
    """

    name = "isotonic"

    def __init__(self, clip: float = 1e-4):
        self.clip = clip

    def fit(self, scores, labels):
        order = np.argsort(scores, kind="stable")
        s = np.asarray(scores, float)[order]
        fitted = pav(np.asarray(labels, float)[order], balanced_weights(labels)[order])
        # keep one point per distinct fitted level (its first and last score) for interpolation
        change = np.flatnonzero(np.diff(fitted)) + 1
        keep = np.unique(np.concatenate([[0, len(s) - 1], change, change - 1]))
        self.x, self.y = s[keep], np.clip(fitted[keep], self.clip, 1 - self.clip)
        return self

    def llr(self, scores):
        p = np.interp(np.asarray(scores, float), self.x, self.y)
        return np.log(p / (1 - p))


def probability(llr, prior: float = 0.5):
    """P(same person | score) for a stated prior share of genuine attempts."""
    return 1 / (1 + np.exp(-(np.asarray(llr) + np.log(prior / (1 - prior)))))


CALIBRATORS = {"naive": Naive, "platt": Platt, "isotonic": Isotonic}


# --- metrics (all at prior 0.5, i.e. both classes weighted equally) ------------------------


def cllr_terms(llr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-pair Cllr loss in bits: log2(1 + e^-LLR) for genuine, log2(1 + e^LLR) for impostor."""
    signed = np.where(labels, -llr, llr)
    return np.logaddexp(0, signed) / np.log(2)


def cllr(llr: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, bool)
    t = cllr_terms(llr, labels)
    return float(0.5 * t[labels].mean() + 0.5 * t[~labels].mean())


def min_cllr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Cllr after the best monotone calibration on the same data: what discrimination allows."""
    iso = Isotonic(clip=1e-12).fit(scores, labels)
    return cllr(iso.llr(scores), labels)


def ece(llr: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of P(same) at prior 0.5, equal-width bins."""
    p = 1 / (1 + np.exp(-llr))
    w = balanced_weights(labels)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = np.bincount(idx, weights=w, minlength=bins)
    gap = np.bincount(idx, weights=w * (p - np.asarray(labels, float)), minlength=bins)
    return float(np.abs(gap).sum() / total.sum())


def brier(llr: np.ndarray, labels: np.ndarray) -> float:
    p = 1 / (1 + np.exp(-llr))
    w = balanced_weights(labels)
    return float((w * (p - np.asarray(labels, float)) ** 2).sum() / w.sum())


# --- person-level bootstrap -----------------------------------------------------------------


def boot_metrics(
    llr_gen: np.ndarray,
    id_gen: np.ndarray,
    llr_imp: np.ndarray,
    id_a: np.ndarray,
    id_b: np.ndarray,
    n_ids: int,
    weights: np.ndarray,
    bins: int = 10,
) -> dict[str, np.ndarray]:
    """Cllr, ECE and Brier on each resample of people (rows of `weights`, shape (boot, n_ids)).

    A genuine pair of person i counts w_i times; an impostor pair of people i and j counts
    w_i * w_j times. Each statistic is summed per person (genuine) or per person pair
    (impostor) once, so every resample is a matrix product.
    """

    def per_person(x):
        return weights @ np.bincount(id_gen, weights=x, minlength=n_ids)

    flat = id_a.astype(np.int64) * n_ids + id_b

    def per_pair(x):
        m = np.bincount(flat, weights=x, minlength=n_ids * n_ids).reshape(n_ids, n_ids)
        return np.einsum("bi,ij,bj->b", weights, m, weights)

    p_gen, p_imp = 1 / (1 + np.exp(-llr_gen)), 1 / (1 + np.exp(-llr_imp))
    g_n, i_n = per_person(np.ones_like(llr_gen)), per_pair(np.ones_like(llr_imp))
    out = {
        "cllr": 0.5 * per_person(cllr_terms(llr_gen, np.ones(len(llr_gen), bool))) / g_n
        + 0.5 * per_pair(cllr_terms(llr_imp, np.zeros(len(llr_imp), bool))) / i_n,
        "brier": 0.5 * per_person((p_gen - 1) ** 2) / g_n + 0.5 * per_pair(p_imp**2) / i_n,
    }
    b_gen = np.minimum((p_gen * bins).astype(int), bins - 1)
    b_imp = np.minimum((p_imp * bins).astype(int), bins - 1)
    gap = np.zeros(len(weights))
    for k in range(bins):
        g, i = b_gen == k, b_imp == k
        gen_gap = per_person(np.where(g, p_gen - 1, 0.0)) / g_n
        imp_gap = per_pair(np.where(i, p_imp, 0.0)) / i_n
        gap += np.abs(0.5 * gen_gap + 0.5 * imp_gap)
    out["ece"] = gap
    return out
