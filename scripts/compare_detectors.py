"""Bootstrap AP intervals and paired differences for detectors already run by eval_wider.py.

Every detector is scored on the same 1,000 resamples of the 3,226 validation images, so a
difference between two detectors is paired. Writes outputs/wider/compare.json. Run on gpu-box.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from faceid_bench.wider_eval import LEVELS, ap_from_curves, bootstrap_ap, bootstrap_weights

CACHE = Path("data/cache/wider")
parser = argparse.ArgumentParser()
parser.add_argument("detectors", nargs="+")
parser.add_argument("--baseline", default="yunet_2026may")
parser.add_argument("--boot", type=int, default=1000)
args = parser.parse_args()


def interval(values: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in np.percentile(values, [2.5, 97.5])]


boots, result = {}, {"n_boot": args.boot, "baseline": args.baseline, "detectors": {}}
for name in dict.fromkeys([args.baseline, *args.detectors]):
    data = np.load(CACHE / f"{name}_curves.npz")
    row = {}
    for level in LEVELS:
        curves, faces = data[f"{level}_curves"].astype(np.float64), data[f"{level}_faces"]
        weights = bootstrap_weights(len(faces), args.boot)  # same seed -> same resamples
        boots[name, level] = bootstrap_ap(curves, faces, weights)
        row[level] = {
            "ap": round(ap_from_curves(curves.sum(axis=0), faces.sum()), 4),
            "ci95": interval(boots[name, level]),
        }
    result["detectors"][name] = row
    print(name, {k: v["ap"] for k, v in row.items()})

for name, row in result["detectors"].items():
    if name == args.baseline:
        continue
    for level in LEVELS:
        diff = boots[name, level] - boots[args.baseline, level]
        lo, hi = interval(diff)
        row[level]["diff_vs_baseline"] = {
            "mean": round(float(diff.mean()), 4),
            "ci95": [lo, hi],
            "significant": bool(lo > 0 or hi < 0),
        }

out = Path("outputs/wider/compare.json")
out.write_text(json.dumps(result, indent=2) + "\n")
print(f"saved {out}")
