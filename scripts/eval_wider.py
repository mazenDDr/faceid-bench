"""Run a detector over WIDER FACE val and score AP (easy / medium / hard). Run on gpu-box.

  python scripts/eval_wider.py yunet_2023mar
Detections and per-image curves are cached in data/cache/wider/ (large, never pulled to the
Mac); APs are appended to outputs/wider/results.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from faceid_bench.detectors import DETECTORS
from faceid_bench.wider_eval import ap_from_curves, image_curves, load_ground_truth

ROOT = Path("data/wider_face")
parser = argparse.ArgumentParser()
parser.add_argument("detector", choices=sorted(DETECTORS))
args = parser.parse_args()

gt = load_ground_truth(ROOT / "eval_tools" / "ground_truth")
out_dir, cache_dir = Path("outputs/wider"), Path("data/cache/wider")
out_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)
cache = cache_dir / f"{args.detector}.npz"

if cache.exists():
    detections = dict(np.load(cache))
    print(f"reusing {cache}")
else:
    detector = DETECTORS[args.detector]()
    detections, start = {}, time.perf_counter()
    for i, key in enumerate(sorted(gt["boxes"])):
        image = cv2.imread(str(ROOT / "WIDER_val" / "images" / f"{key}.jpg"))
        detections[key] = detector(image)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(gt['boxes'])} images, {time.perf_counter() - start:.0f} s")
    np.savez_compressed(cache, **detections)

predictions = {k: np.column_stack([v[:, :4], v[:, -1]]) for k, v in detections.items()}
curves = image_curves(predictions, gt)
# per-image curves let compare_detectors.py bootstrap AP and paired differences later
np.savez_compressed(
    cache_dir / f"{args.detector}_curves.npz",
    **{f"{level}_curves": c.astype(np.float32) for level, (c, _) in curves.items()},
    **{f"{level}_faces": f for level, (_, f) in curves.items()},
)
aps = {level: round(ap_from_curves(c.sum(axis=0), f.sum()), 4) for level, (c, f) in curves.items()}
row = {"detector": args.detector, **{f"ap_{k}": v for k, v in aps.items()}}
row["detections"] = int(sum(len(v) for v in detections.values()))
with (out_dir / "results.jsonl").open("a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row))
