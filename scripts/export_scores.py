"""Histogram of test-people similarity scores for the balanced pipeline, for the results page.

Counts per 0.005-wide bin from -0.3 to 1.0, genuine and impostor separately, so the page can
compute the exact true-accept and false-accept rates at any bin edge. Run on the GPU machine.
Writes outputs/page/scores.json.
"""

import json
from pathlib import Path

import numpy as np

from faceid_bench.data import identity_split, lfw_images
from faceid_bench.verify import PairScores

MODEL = "r50_w600k__scrfd_10g_kps@320"
edges = np.round(np.arange(-0.3, 1.0 + 1e-9, 0.005), 4)
emb = np.load(f"data/cache/lfw/{MODEL}.npz")
index = {p: i for i, p in enumerate(emb["paths"])}
people = identity_split(lfw_images())["test"]
names = sorted(people)
idx = np.array([index[p] for n in names for p in people[n]])
ident = np.array([k for k, n in enumerate(names) for _ in people[n]])
s = PairScores(emb["embeddings"][idx], ident)
for label, scores in (("genuine", s.genuine), ("impostor", s.impostor)):
    assert scores.min() >= edges[0] and scores.max() <= edges[-1], (
        label,
        scores.min(),
        scores.max(),
    )
out = {
    "model": MODEL,
    "edges": edges.tolist(),
    "genuine": np.histogram(s.genuine, edges)[0].tolist(),
    "impostor": np.histogram(s.impostor, edges)[0].tolist(),
    "people": int(s.n_ids),
}
Path("outputs/page").mkdir(parents=True, exist_ok=True)
Path("outputs/page/scores.json").write_text(json.dumps(out) + "\n")
print(
    {
        k: (sum(v) if isinstance(v, list) and k != "edges" else v)
        for k, v in out.items()
        if k != "edges"
    }
)
