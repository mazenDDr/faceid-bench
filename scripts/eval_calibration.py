"""Fit score -> LLR calibrators on dev people and score them on test people. Run on gpu-box.

Writes outputs/calibration/compare.json: Cllr / ECE / Brier at prior 0.5 with 95% intervals
from resampling test people (shared across calibrators, so differences are paired), minimum
Cllr (the floor set by the model's discrimination), reliability bins, and the calibrated LLR
at the FAR thresholds from the verification step.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from faceid_bench.calibrate import CALIBRATORS, boot_metrics, brier, cllr, ece, min_cllr
from faceid_bench.data import LFW_LABEL_ERRORS, LFW_NAME_COLLISIONS, identity_split, lfw_images
from faceid_bench.verify import PairScores, identity_weights, threshold_at_far

CACHE = Path("data/cache/lfw")
parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="+")
parser.add_argument("--boot", type=int, default=200)
parser.add_argument("--out-name", default="compare")
parser.add_argument("--dev-impostors", type=int, default=1_000_000)
parser.add_argument(
    "--drop-label-errors",
    action="store_true",
    help="sensitivity check: drop mislabelled images and name collisions",
)
args = parser.parse_args()
suffix = "_clean" if args.drop_label_errors else ""


def interval(v):
    return [round(float(x), 5) for x in np.percentile(v, [2.5, 97.5])]


def side_scores(emb, index, people):
    names = sorted(people)
    idx = np.array([index[p] for n in names for p in people[n]])
    ident = np.array([k for k, n in enumerate(names) for _ in people[n]])
    return PairScores(emb[idx], ident)


def pairs_of(s: PairScores, mask):
    rows, cols = np.nonzero(mask)
    return s.sim[rows, cols], s.identity[rows], s.identity[cols]


split = identity_split(
    lfw_images(),
    drop=set(LFW_LABEL_ERRORS) if args.drop_label_errors else frozenset(),
    drop_identities=set(LFW_NAME_COLLISIONS) if args.drop_label_errors else frozenset(),
)
result = {"prior": 0.5, "n_boot": args.boot, "models": {}}
for name in args.models:
    data = np.load(CACHE / f"{name}.npz")
    index = {p: i for i, p in enumerate(data["paths"])}
    dev = side_scores(data["embeddings"], index, split["dev"])
    test = side_scores(data["embeddings"], index, split["test"])

    # one generator per model, so a model's dev sample does not depend on which ran before it
    rng = np.random.default_rng(0)
    sample = rng.choice(len(dev.impostor), args.dev_impostors, replace=False)
    fit_s = np.concatenate([dev.genuine, dev.impostor[sample]])
    fit_y = np.concatenate([np.ones(len(dev.genuine), bool), np.zeros(len(sample), bool)])

    s_gen, id_gen, _ = pairs_of(test, test.genuine_mask)
    s_imp, id_a, id_b = pairs_of(test, test.impostor_mask)
    weights = identity_weights(test.n_ids, args.boot)
    labels = np.concatenate([np.ones(len(s_gen), bool), np.zeros(len(s_imp), bool)])
    scores = np.concatenate([s_gen, s_imp])

    row = {"min_cllr_test": round(min_cllr(scores, labels), 5), "calibrators": {}}
    boots = {}
    for cal_name, cal_cls in CALIBRATORS.items():
        cal = cal_cls().fit(fit_s, fit_y)
        llr_gen, llr_imp = cal.llr(s_gen), cal.llr(s_imp)
        boots[cal_name] = boot_metrics(llr_gen, id_gen, llr_imp, id_a, id_b, test.n_ids, weights)
        llr = np.concatenate([llr_gen, llr_imp])
        p = 1 / (1 + np.exp(-llr))
        bins = np.minimum((p * 10).astype(int), 9)
        w = np.where(labels, 0.5 / len(s_gen), 0.5 / len(s_imp))
        total = np.bincount(bins, weights=w, minlength=10)
        with np.errstate(invalid="ignore"):
            predicted = np.bincount(bins, weights=w * p, minlength=10) / total
            observed = np.bincount(bins, weights=w * labels, minlength=10) / total
        reliability = {  # empty bins -> null
            "mean_predicted": [None if total[k] == 0 else float(predicted[k]) for k in range(10)],
            "observed": [None if total[k] == 0 else float(observed[k]) for k in range(10)],
            "weight": total.tolist(),
        }
        point = {"cllr": cllr(llr, labels), "ece": ece(llr, labels), "brier": brier(llr, labels)}
        entry = {
            **{
                m: {"point": round(point[m], 5), "ci95": interval(v)}
                for m, v in boots[cal_name].items()
            },
            "reliability": reliability,
            "llr_at_dev_far_threshold": {
                f"{far:g}": round(
                    float(cal.llr(np.array([threshold_at_far(dev.impostor, far)]))[0]), 3
                )
                for far in (1e-3, 1e-4, 1e-5)
            },
        }
        if cal_name == "platt":
            entry["params"] = {"a": round(float(cal.a), 4), "b": round(float(cal.b), 4)}
        row["calibrators"][cal_name] = entry
    for cal_name in ("platt", "naive"):
        diff = boots["isotonic"]["cllr"] - boots[cal_name]["cllr"]
        lo, hi = interval(diff)
        row[f"cllr_isotonic_minus_{cal_name}"] = {
            "mean": round(float(diff.mean()), 5),
            "ci95": [lo, hi],
            "significant": bool(lo > 0 or hi < 0),
        }
    result["models"][name] = row
    print(
        name,
        "minCllr",
        row["min_cllr_test"],
        {k: (v["cllr"]["point"], v["ece"]["point"]) for k, v in row["calibrators"].items()},
    )

out = Path(f"outputs/calibration/{args.out_name}{suffix}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(f"saved {out}")
