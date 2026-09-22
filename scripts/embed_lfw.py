"""Detect, align and embed every LFW image with each face model. Run on gpu-box.

The face used is the detection (score >= 0.5) nearest the image centre, since LFW images are
centred on the named person. Images with no such face fall back to the median landmarks of
all detected faces; how many did is recorded. Caches go to data/cache/lfw/ (never pulled).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from faceid_bench.align import align
from faceid_bench.data import DATA, lfw_images
from faceid_bench.detectors import SCRFD
from faceid_bench.embedders import EMBEDDERS, ArcFaceONNX

ROOT = DATA / "lfw" / "lfw"
CACHE = Path("data/cache/lfw")
CUDA = ("CUDAExecutionProvider", "CPUExecutionProvider")
parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="+", choices=sorted(EMBEDDERS))
parser.add_argument(
    "--detector",
    default="scrfd_10g_kps",
    choices=["scrfd_10g_kps", "scrfd_10g_kps@320", "scrfd_500m_kps@320"],
    help="non-default detectors tag the cache as <model>__<detector>",
)
args = parser.parse_args()
DETECTOR_FILES = {
    "scrfd_10g_kps": ("buffalo_l/det_10g.onnx", 640),
    "scrfd_10g_kps@320": ("buffalo_l/det_10g.onnx", 320),
    "scrfd_500m_kps@320": ("buffalo_s/det_500m.onnx", 320),
}
tag = "" if args.detector == "scrfd_10g_kps" else f"__{args.detector}"
CACHE.mkdir(parents=True, exist_ok=True)

paths = [p for ps in lfw_images(ROOT).values() for p in ps]
det_file = CACHE / f"detections_{args.detector}.npz"
if det_file.exists():
    saved = np.load(det_file)
    rows, found = saved["rows"], saved["found"]
else:
    path, size = DETECTOR_FILES[args.detector]
    detector = SCRFD(path, size=size, providers=CUDA)
    rows, found, start = (
        np.zeros((len(paths), 15), np.float32),
        np.zeros(len(paths), bool),
        time.time(),
    )
    for i, rel in enumerate(paths):
        image = cv2.imread(str(ROOT / rel))
        det = detector(image)
        det = det[det[:, 14] >= 0.5]
        if len(det):
            centre = np.array([image.shape[1], image.shape[0]]) / 2
            dist = np.linalg.norm(det[:, :2] + det[:, 2:4] / 2 - centre, axis=1)
            rows[i], found[i] = det[np.argmin(dist)], True
    rows[~found, 4:14] = np.median(rows[found, 4:14], axis=0)
    rows[~found, :4] = np.median(rows[found, :4], axis=0)
    np.savez_compressed(det_file, paths=np.array(paths), rows=rows, found=found)
    print(f"detected {found.sum()}/{len(paths)} in {time.time() - start:.0f} s")

summary = {"detector": args.detector, "images": len(paths), "no_face_fallback": int((~found).sum())}
for name in args.models:
    model = EMBEDDERS[name](CUDA)
    start, embeddings = time.time(), []
    if isinstance(model, ArcFaceONNX):
        for i in range(0, len(paths), 256):
            batch = [
                align(cv2.imread(str(ROOT / rel)), row[4:14])
                for rel, row in zip(paths[i : i + 256], rows[i : i + 256], strict=True)
            ]
            embeddings.append(model.embed_crops(np.stack(batch)))
        embeddings = np.concatenate(embeddings)
    else:
        embeddings = np.stack(
            [model(cv2.imread(str(ROOT / rel)), row) for rel, row in zip(paths, rows, strict=True)]
        )
    np.savez_compressed(CACHE / f"{name}{tag}.npz", paths=np.array(paths), embeddings=embeddings)
    summary[name] = {"dim": int(embeddings.shape[1]), "seconds": round(time.time() - start, 1)}
    print(name, summary[name])

Path("outputs/verification").mkdir(parents=True, exist_ok=True)
Path(f"outputs/verification/embed_summary{tag}.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary))
