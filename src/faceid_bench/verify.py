"""Verification metrics: the official LFW 10-fold accuracy and a Face ID-style operating point.

Face ID-style: pick the similarity threshold on dev people so that a target share of impostor
pairs is accepted (FAR), then report on test people the share of genuine pairs accepted (TAR)
and the FAR actually reached. Intervals resample people, not pairs, because all pairs of one
person share that person's photos.
"""

from __future__ import annotations

import numpy as np


def lfw_accuracy(
    emb_a: np.ndarray, emb_b: np.ndarray, same: np.ndarray, folds: np.ndarray
) -> tuple[float, float]:
    """Official protocol: best squared-L2 threshold on 9 folds, accuracy on the held-out fold.

    Thresholds 0.00, 0.01, ... 3.99 as in the common evaluation code (and OpenCV Zoo's).
    """
    dist = np.sum((emb_a - emb_b) ** 2, axis=1)
    thresholds = np.arange(0, 4, 0.01)
    accept = dist[None, :] < thresholds[:, None]  # (thresholds, pairs)
    correct = accept == same[None, :]
    accs = []
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        best = np.argmax(correct[:, train].mean(axis=1))
        accs.append(correct[best, test].mean())
    return float(np.mean(accs)), float(np.std(accs))


def threshold_at_far(impostor_scores: np.ndarray, far: float) -> float:
    """Smallest threshold (accept if score > t) whose impostor accept rate is <= far."""
    return float(np.quantile(impostor_scores, 1 - far, method="higher"))


class PairScores:
    """All-vs-all cosine scores for one set of people, kept per identity for the bootstrap."""

    def __init__(self, embeddings: np.ndarray, identity: np.ndarray):
        self.identity = np.asarray(identity)
        self.n_ids = int(self.identity.max()) + 1
        self.sizes = np.bincount(self.identity, minlength=self.n_ids)
        sim = embeddings @ embeddings.T
        upper = np.triu(np.ones(sim.shape, dtype=bool), k=1)
        same = self.identity[:, None] == self.identity[None, :]
        self.sim = sim
        self.genuine_mask = upper & same
        self.impostor_mask = upper & ~same
        self.genuine = sim[self.genuine_mask]
        self.impostor = sim[self.impostor_mask]

    def rates(self, t: float) -> dict[str, float]:
        return {
            "tar": float((self.genuine > t).mean()),
            "far": float((self.impostor > t).mean()),
            "false_accepts": int((self.impostor > t).sum()),
        }

    def per_identity(self, t: float) -> dict[str, np.ndarray]:
        """Counts needed to recompute TAR / FAR on resampled people at threshold t."""
        rows, cols = np.nonzero(self.genuine_mask)
        genuine_ok = np.bincount(
            self.identity[rows], weights=self.sim[rows, cols] > t, minlength=self.n_ids
        )
        genuine_all = self.sizes * (self.sizes - 1) / 2
        rows, cols = np.nonzero(self.impostor_mask & (self.sim > t))
        a, b = self.identity[rows], self.identity[cols]
        accepted = np.zeros((self.n_ids, self.n_ids))
        np.add.at(accepted, (a, b), 1)
        accepted = accepted + accepted.T  # count both orders, like the denominator below
        return {"genuine_ok": genuine_ok, "genuine_all": genuine_all, "impostor_ok": accepted}


def bootstrap_rates(counts: dict[str, np.ndarray], sizes: np.ndarray, weights: np.ndarray):
    """TAR and FAR for each row of identity resampling counts `weights` (n_boot, n_ids).

    A person drawn twice contributes no impostor pairs with their own copy.
    """
    tar = weights @ counts["genuine_ok"] / (weights @ counts["genuine_all"])
    accepted = np.einsum("bi,ij,bj->b", weights, counts["impostor_ok"], weights)
    total = (weights @ sizes) ** 2 - weights**2 @ sizes**2
    return tar, accepted / total


def identity_weights(n_ids: int, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_ids, np.full(n_ids, 1 / n_ids), size=n_boot).astype(np.float64)


def auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """P(a random genuine pair scores above a random impostor pair); ties count half."""
    both = np.concatenate([genuine, impostor])
    _, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]  # average rank within ties
    n_g, n_i = len(genuine), len(impostor)
    return float((ranks[:n_g].sum() - n_g * (n_g + 1) / 2) / (n_g * n_i))
