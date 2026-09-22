"""Laya as the "same person?" decision engine vs Platt on the cosine score. Run on gpu-box.

Both read the same ResNet-50 similarity of the same LFW pairs. Laya answers a `noul` question
from a JSON state (three variants); "laya+platt" recalibrates Laya's output on dev people.
Thresholds for TAR at FAR 1e-3 are set on the dev sample. Writes outputs/laya/compare.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from faceid_bench.calibrate import Platt, boot_metrics, cllr, ece
from faceid_bench.data import identity_split, lfw_images
from faceid_bench.laya_decision import VARIANTS, LayaJudge, state
from faceid_bench.verify import PairScores, auc, identity_weights, threshold_at_far

CACHE = Path("data/cache/lfw")
parser = argparse.ArgumentParser()
parser.add_argument("--test", type=int, nargs=2, default=[2000, 20000], help="genuine impostor")
parser.add_argument("--dev", type=int, nargs=2, default=[1000, 10000], help="genuine impostor")
parser.add_argument("--boot", type=int, default=200)
parser.add_argument(
    "--model-id", default="convaiinnovations/laya", help="HF id or local checkpoint"
)
parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
parser.add_argument("--out-name", default="compare")
args = parser.parse_args()
rng = np.random.default_rng(0)

emb = np.load(CACHE / "r50_w600k.npz")
det = np.load(CACHE / "detections_scrfd_10g_kps.npz")
index = {p: i for i, p in enumerate(emb["paths"])}
split = identity_split(lfw_images())


def sample(people, n_gen, n_imp):
    """Random genuine and impostor pairs among `people`; returns global image indices + ids."""
    names = sorted(people)
    idx = np.array([index[p] for n in names for p in people[n]])
    ident = np.array([k for k, n in enumerate(names) for _ in people[n]])
    s = PairScores(emb["embeddings"][idx], ident)
    out = {}
    for label, mask, n in (("gen", s.genuine_mask, n_gen), ("imp", s.impostor_mask, n_imp)):
        r, c = np.nonzero(mask)
        pick = rng.choice(len(r), n, replace=False)
        r, c = r[pick], c[pick]
        out[label] = (idx[r], idx[c], s.sim[r, c], ident[r], ident[c])
    out["n_ids"], out["dev_all"] = s.n_ids, s
    return out


dev, test = sample(split["dev"], *args.dev), sample(split["test"], *args.test)
judge = LayaJudge(args.model_id)


def laya_llr(pairs, variant):
    a, b, sim = pairs[0], pairs[1], pairs[2]
    states = [
        state(variant, s, det["rows"][i], det["rows"][j]) for i, j, s in zip(a, b, sim, strict=True)
    ]
    start = time.perf_counter()
    p = np.clip(judge.noul(states), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)), (time.perf_counter() - start) / len(states) * 1e3


# Platt on the cosine, fit exactly as in the calibration step (all dev genuine + 1M impostors)
d = dev["dev_all"]
pick = rng.choice(len(d.impostor), 1_000_000, replace=False)
platt = Platt().fit(
    np.concatenate([d.genuine, d.impostor[pick]]),
    np.concatenate([np.ones(len(d.genuine), bool), np.zeros(len(pick), bool)]),
)

methods = {
    "platt_cosine": {
        "dev": (platt.llr(dev["gen"][2]), platt.llr(dev["imp"][2])),
        "test": (platt.llr(test["gen"][2]), platt.llr(test["imp"][2])),
        "ms_per_decision": None,
    }
}
for v in args.variants:
    dg, _ = laya_llr(dev["gen"], v)
    di, _ = laya_llr(dev["imp"], v)
    tg, ms = laya_llr(test["gen"], v)
    ti, _ = laya_llr(test["imp"], v)
    methods[f"laya_{v}"] = {"dev": (dg, di), "test": (tg, ti), "ms_per_decision": ms}
    recal = Platt().fit(
        np.concatenate([dg, di]), np.concatenate([np.ones(len(dg), bool), np.zeros(len(di), bool)])
    )
    methods[f"laya_{v}+platt"] = {
        "dev": (recal.llr(dg), recal.llr(di)),
        "test": (recal.llr(tg), recal.llr(ti)),
        "ms_per_decision": ms,
    }
    print(v, f"{ms:.1f} ms/decision", flush=True)


weights = identity_weights(test["n_ids"], args.boot)
id_gen, id_a, id_b = test["gen"][3], test["imp"][3], test["imp"][4]
result = {
    "model": args.model_id,
    "test_pairs": args.test,
    "dev_pairs": args.dev,
    "far": 1e-3,
    "methods": {},
}
boots = {}
for name, m in methods.items():
    (dg, di), (tg, ti) = m["dev"], m["test"]
    t = threshold_at_far(di, 1e-3)
    labels = np.concatenate([np.ones(len(tg), bool), np.zeros(len(ti), bool)])
    llr = np.concatenate([tg, ti])
    b = boot_metrics(tg, id_gen, ti, id_a, id_b, test["n_ids"], weights)
    tar_b = (weights @ np.bincount(id_gen, weights=tg > t, minlength=test["n_ids"])) / (
        weights @ np.bincount(id_gen, minlength=test["n_ids"])
    )
    boots[name] = {"cllr": b["cllr"], "tar": tar_b}
    result["methods"][name] = {
        "auc": round(auc(tg, ti), 5),
        "tar_at_far_1e-3": round(float((tg > t).mean()), 5),
        "tar_ci95": [round(float(x), 5) for x in np.percentile(tar_b, [2.5, 97.5])],
        "test_far": round(float((ti > t).mean()), 6),
        "cllr": round(cllr(llr, labels), 5),
        "cllr_ci95": [round(float(x), 5) for x in np.percentile(b["cllr"], [2.5, 97.5])],
        "ece": round(ece(llr, labels), 5),
        "ms_per_decision_rtx": None
        if m["ms_per_decision"] is None
        else round(m["ms_per_decision"], 2),
    }
for name in methods:
    if name == "platt_cosine":
        continue
    for metric in ("cllr", "tar"):
        diff = boots[name][metric] - boots["platt_cosine"][metric]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        result["methods"][name][f"{metric}_minus_platt_cosine"] = {
            "mean": round(float(diff.mean()), 5),
            "ci95": [round(float(lo), 5), round(float(hi), 5)],
            "significant": bool(lo > 0 or hi < 0),
        }
out = Path(f"outputs/laya/{args.out_name}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: {m: v[m] for m in ("auc", "tar_at_far_1e-3", "cllr", "ece")}
            for k, v in result["methods"].items()
        },
        indent=1,
    )
)
