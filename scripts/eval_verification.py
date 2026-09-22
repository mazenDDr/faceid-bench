"""Score face embedders on LFW. Run on gpu-box after embed_lfw.py.

1. Official 10-fold accuracy on pairs.txt (comparable with published numbers).
2. Face ID-style: thresholds fixed on dev people at FAR 1e-3 / 1e-4 / 1e-5, then TAR and the
   FAR actually reached on test people, with 95% intervals from resampling test people (the
   same resamples for every model, so differences are paired).
Writes outputs/verification/compare.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from faceid_bench.data import DATA, LFW_LABEL_ERRORS, identity_split, lfw_images, read_lfw_pairs
from faceid_bench.verify import (
    PairScores,
    bootstrap_rates,
    identity_weights,
    lfw_accuracy,
    threshold_at_far,
)

CACHE = Path("data/cache/lfw")
FARS = (1e-3, 1e-4, 1e-5)
parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="+")
parser.add_argument("--baseline", default="mbf_w600k")
parser.add_argument("--boot", type=int, default=1000)
parser.add_argument("--out-name", default="compare")
parser.add_argument(
    "--drop-label-errors", action="store_true", help="sensitivity check: remove LFW_LABEL_ERRORS"
)
args = parser.parse_args()
suffix = "_clean" if args.drop_label_errors else ""


def interval(values):
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


pairs = read_lfw_pairs(DATA / "lfw" / "pairs.txt")
split = identity_split(
    lfw_images(), drop=set(LFW_LABEL_ERRORS) if args.drop_label_errors else frozenset()
)
result = {"fars": list(FARS), "n_boot": args.boot, "baseline": args.baseline, "models": {}}
boots, weights = {}, None
for name in dict.fromkeys([args.baseline, *args.models]):
    data = np.load(CACHE / f"{name}.npz")
    index = {p: i for i, p in enumerate(data["paths"])}
    emb = data["embeddings"]

    ia = np.array([index[a] for _, a, _, _ in pairs])
    ib = np.array([index[b] for _, _, b, _ in pairs])
    same = np.array([s for *_, s in pairs])
    folds = np.array([f for f, *_ in pairs])
    acc, acc_std = lfw_accuracy(emb[ia], emb[ib], same, folds)
    row = {"lfw_10fold": {"accuracy": round(acc, 5), "std": round(acc_std, 5)}}

    sides = {}
    for side, people in split.items():
        names = sorted(people)
        idx = np.array([index[p] for n in names for p in people[n]])
        ident = np.array([k for k, n in enumerate(names) for _ in people[n]])
        sides[side] = PairScores(emb[idx], ident)
    dev, test = sides["dev"], sides["test"]
    if weights is None:
        weights = identity_weights(test.n_ids, args.boot)
        result["test_people"] = int(test.n_ids)
        result["test_pairs"] = {"genuine": len(test.genuine), "impostor": len(test.impostor)}

    row["operating_points"] = {}
    for far in FARS:
        t = threshold_at_far(dev.impostor, far)
        point = test.rates(t)
        tar_b, far_b = bootstrap_rates(test.per_identity(t), test.sizes.astype(float), weights)
        boots[name, far] = tar_b
        oracle = test.rates(threshold_at_far(test.impostor, far))
        row["operating_points"][f"{far:g}"] = {
            "threshold_from_dev": round(t, 5),
            "test_tar": round(point["tar"], 5),
            "test_tar_ci95": [round(v, 5) for v in interval(tar_b)],
            "test_far": point["far"],
            "test_far_ci95": interval(far_b),
            "test_false_accepts": point["false_accepts"],
            "test_tar_at_test_threshold": round(oracle["tar"], 5),
        }
    result["models"][name] = row
    print(name, row["lfw_10fold"], {k: v["test_tar"] for k, v in row["operating_points"].items()})

for name, row in result["models"].items():
    if name == args.baseline:
        continue
    for far in FARS:
        diff = boots[name, far] - boots[args.baseline, far]
        lo, hi = interval(diff)
        row["operating_points"][f"{far:g}"]["tar_diff_vs_baseline"] = {
            "mean": round(float(diff.mean()), 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "significant": bool(lo > 0 or hi < 0),
        }

out = Path(f"outputs/verification/{args.out_name}{suffix}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(f"saved {out}")
