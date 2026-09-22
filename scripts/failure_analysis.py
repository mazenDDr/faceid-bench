"""Trace every mistake of the balanced pipeline to the stage that broke. Run on gpu-box.

Failures are taken at the FAR 1e-5 threshold fixed on dev people. Each image gets quality
features; the attribution thresholds come from the dev distribution (printed with
--dev-quantiles), never from the test failures. Also checks whether the accurate configuration
recovers each failure, which separates detection problems from embedding ones.
Writes outputs/failures/analysis.json.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from faceid_bench.align import align
from faceid_bench.data import DATA, LFW_LABEL_ERRORS, identity_split, lfw_images
from faceid_bench.failures import features, pair_stage
from faceid_bench.verify import PairScores, threshold_at_far

CACHE = Path("data/cache/lfw")
ROOT = DATA / "lfw" / "lfw"
BALANCED = ("r50_w600k__scrfd_10g_kps@320", "scrfd_10g_kps@320")
ACCURATE = ("r50_w600k", "scrfd_10g_kps")
parser = argparse.ArgumentParser()
parser.add_argument("--far", type=float, default=1e-5)
parser.add_argument("--dev-quantiles", action="store_true")
args = parser.parse_args()
split = identity_split(lfw_images())


def side(embeddings, paths_index, people):
    names = sorted(people)
    idx = np.array([paths_index[p] for n in names for p in people[n]])
    ident = np.array([k for k, n in enumerate(names) for _ in people[n]])
    paths = [p for n in names for p in people[n]]
    return PairScores(embeddings[idx], ident), idx, paths


def load(config):
    emb = np.load(CACHE / f"{config[0]}.npz")
    det = np.load(CACHE / f"detections_{config[1]}.npz")
    index = {p: i for i, p in enumerate(emb["paths"])}
    return emb["embeddings"], det["rows"], det["found"], index


embeddings, rows, found, index = load(BALANCED)
dev, dev_idx, dev_paths = side(embeddings, index, split["dev"])
test, test_idx, test_paths = side(embeddings, index, split["test"])


def image_features(global_idx, path):
    image = cv2.imread(str(ROOT / path))
    crop = align(image, rows[global_idx][4:14])
    return features(rows[global_idx], bool(found[global_idx]), crop)


if args.dev_quantiles:
    sample = np.random.default_rng(0).choice(len(dev_paths), 800, replace=False)
    values = [image_features(dev_idx[i], dev_paths[i]) for i in sample]
    print(
        json.dumps(
            {
                key: {
                    f"p{q}": round(float(np.percentile([v[key] for v in values], q)), 2)
                    for q in (1, 5, 10, 50, 90, 99)
                }
                for key in (
                    "face_width_px",
                    "det_score",
                    "roll_deg",
                    "yaw_proxy",
                    "sharpness",
                    "brightness",
                )
            },
            indent=2,
        )
    )
    raise SystemExit

threshold = threshold_at_far(dev.impostor, args.far)
rows_gen, cols_gen = np.nonzero(test.genuine_mask)
rows_imp, cols_imp = np.nonzero(test.impostor_mask)
sim_gen = test.sim[rows_gen, cols_gen]
sim_imp = test.sim[rows_imp, cols_imp]
false_reject = np.flatnonzero(sim_gen <= threshold)
false_accept = np.flatnonzero(sim_imp > threshold)

# the accurate configuration, for the same pairs
acc_emb, acc_rows, acc_found, acc_index = load(ACCURATE)
acc_test, _, _ = side(acc_emb, acc_index, split["test"])
acc_threshold = threshold_at_far(side(acc_emb, acc_index, split["dev"])[0].impostor, args.far)

cache: dict[int, dict] = {}


def feats(i):
    if i not in cache:
        cache[i] = image_features(test_idx[i], test_paths[i])
    return cache[i]


def analyse(pair_rows, pair_cols, picks, genuine):
    stages, recovered, examples = Counter(), 0, []
    for p in picks:
        a, b = int(pair_rows[p]), int(pair_cols[p])
        label_error = test_paths[a] in LFW_LABEL_ERRORS or test_paths[b] in LFW_LABEL_ERRORS
        stage = pair_stage(feats(a), feats(b), label_error)
        stages[stage] += 1
        acc_sim = float(acc_test.sim[a, b])
        fixed = (acc_sim > acc_threshold) if genuine else (acc_sim <= acc_threshold)
        recovered += int(fixed)
        examples.append(
            {
                "a": test_paths[a],
                "b": test_paths[b],
                "similarity": round(float(test.sim[a, b]), 4),
                "similarity_accurate": round(acc_sim, 4),
                "stage": stage,
                "fixed_by_accurate": bool(fixed),
                "features_a": {
                    k: round(v, 2) if isinstance(v, float) else v for k, v in feats(a).items()
                },
                "features_b": {
                    k: round(v, 2) if isinstance(v, float) else v for k, v in feats(b).items()
                },
            }
        )
    examples.sort(key=lambda e: e["similarity"], reverse=not genuine)
    return {
        "count": len(picks),
        "stages": dict(stages.most_common()),
        "fixed_by_accurate_config": recovered,
        "examples": examples[:15],
    }


result = {
    "far_target": args.far,
    "threshold_from_dev": round(threshold, 5),
    "test_pairs": {"genuine": int(len(sim_gen)), "impostor": int(len(sim_imp))},
    "false_rejects": analyse(rows_gen, cols_gen, false_reject, True),
    "false_accepts": analyse(rows_imp, cols_imp, false_accept, False),
}

# how often each quality bucket fails, over all genuine pairs of the sampled images
sample = np.random.default_rng(1).choice(
    len(false_reject), min(len(false_reject), 400), replace=False
)
result["images_measured"] = len(cache)
out = Path("outputs/failures/analysis.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in result.items() if k != "false_rejects" and k != "false_accepts"}, indent=2
    )
)
for key in ("false_rejects", "false_accepts"):
    print(
        key,
        result[key]["count"],
        result[key]["stages"],
        "fixed by accurate:",
        result[key]["fixed_by_accurate_config"],
    )
