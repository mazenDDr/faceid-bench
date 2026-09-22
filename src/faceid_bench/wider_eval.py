"""WIDER FACE average precision, following the official protocol.

Same steps as the official MATLAB evaluation and its common Python port (used by OpenCV Zoo):
scores are min-max normalised over the whole dataset, each image is matched at IoU 0.5 with the
+1 pixel box convention, faces outside the easy/medium/hard subset are ignored (a detection on
them is neither a hit nor a false positive), precision/recall are read at 1,000 score thresholds,
and AP is the VOC all-point area. Only the per-image loops are vectorised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

THRESH_NUM = 1000
LEVELS = ("easy", "medium", "hard")


def load_ground_truth(gt_dir: str | Path) -> dict[str, object]:
    """Boxes for every face plus the subset indices, keyed by 'event/image' (no extension)."""
    from scipy.io import loadmat

    gt_dir = Path(gt_dir)
    full = loadmat(gt_dir / "wider_face_val.mat")
    boxes, subsets = {}, {level: {} for level in LEVELS}
    levels = {level: loadmat(gt_dir / f"wider_{level}_val.mat")["gt_list"] for level in LEVELS}
    for i, event in enumerate(full["event_list"][:, 0]):
        event_name = str(event[0])
        for j, file in enumerate(full["file_list"][i, 0][:, 0]):
            key = f"{event_name}/{file[0]}"
            boxes[key] = full["face_bbx_list"][i, 0][j, 0].astype(np.float64)
            for level in LEVELS:
                subsets[level][key] = levels[level][i, 0][j, 0].reshape(-1).astype(np.int64)
    return {"boxes": boxes, "subsets": subsets}


def bbox_overlaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between x1y1x2y2 boxes with the +1 pixel convention of the official code."""
    lt = np.maximum(a[:, None, :2], b[:, :2])
    rb = np.minimum(a[:, None, 2:4], b[:, 2:4])
    inter = np.prod(rb - lt + 1, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, :2] + 1, axis=1)
    area_b = np.prod(b[:, 2:4] - b[:, :2] + 1, axis=1)
    return inter / (area_a[:, None] + area_b - inter)


def image_eval(pred: np.ndarray, gt: np.ndarray, keep: np.ndarray, iou: float = 0.5):
    """Greedy matching in score order. Returns (recalled faces so far, proposal counted?) per box.

    `keep` is 1 for faces in the subset. A detection whose best face is outside the subset is
    dropped from the proposals; the best face is used even if another detection already took it,
    exactly as in the official code.
    """
    p = pred[:, :4].copy()
    p[:, 2:] += p[:, :2]
    g = gt.copy()
    g[:, 2:] += g[:, :2]
    overlaps = bbox_overlaps(p, g)
    recalled = np.zeros(len(g))
    proposal = np.ones(len(p))
    pred_recall = np.zeros(len(p))
    count = 0
    for h in range(len(p)):
        best = overlaps[h].argmax()
        if overlaps[h, best] >= iou:
            if keep[best] == 0:
                recalled[best] = -1
                proposal[h] = -1
            elif recalled[best] == 0:
                recalled[best] = 1
                count += 1
        pred_recall[h] = count
    return pred_recall, proposal


def image_pr(scores: np.ndarray, proposal: np.ndarray, pred_recall: np.ndarray) -> np.ndarray:
    """(proposals, recalled faces) at each of the 1,000 thresholds; scores sorted descending."""
    thresh = 1 - (np.arange(THRESH_NUM) + 1) / THRESH_NUM
    last = (scores[:, None] >= thresh[None, :]).sum(axis=0) - 1
    proposals = np.cumsum(proposal == 1)
    out = np.zeros((THRESH_NUM, 2))
    hit = last >= 0
    out[hit, 0] = proposals[last[hit]]
    out[hit, 1] = pred_recall[last[hit]]
    return out


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def prepare(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Round like the reference runner, fill empty images, sort by score, normalise scores.

    Predictions are (n, 5) arrays of x, y, w, h, score keyed like the ground truth.
    """
    out = {}
    for key, det in predictions.items():
        det = np.asarray(det, dtype=np.float64).reshape(-1, 5)
        if len(det) == 0:
            det = np.array([[10, 10, 20, 20, 0.002]])
        det = np.column_stack([np.around(det[:, :4], 1), np.around(det[:, 4], 3)])
        out[key] = det[np.argsort(-det[:, 4], kind="stable")]
    scores = np.concatenate([d[:, 4] for d in out.values()])
    lo, hi = min(scores.min(), 1.0), max(scores.max(), 0.0)
    for det in out.values():
        det[:, 4] = (det[:, 4] - lo) / (hi - lo)
    return out


def image_curves(predictions: dict[str, np.ndarray], gt: dict[str, object], iou: float = 0.5):
    """Per-image (proposals, recalled) curves and face counts for each subset.

    Returns {level: (curves (images, 1000, 2), faces (images,))}, images in sorted key order.
    Summing over images and calling `ap_from_curves` gives the official AP; resampling images
    first gives a bootstrap interval.
    """
    missing = set(gt["boxes"]) - set(predictions)
    if missing:
        raise ValueError(f"{len(missing)} images have no prediction entry, e.g. {min(missing)}")
    preds = prepare(predictions)
    keys = sorted(gt["boxes"])
    out = {}
    for level in LEVELS:
        curves = np.zeros((len(keys), THRESH_NUM, 2))
        faces = np.zeros(len(keys))
        for n, key in enumerate(keys):
            boxes, keep_index = gt["boxes"][key], gt["subsets"][level][key]
            faces[n] = len(keep_index)
            if len(boxes) == 0:
                continue
            keep = np.zeros(len(boxes))
            keep[keep_index - 1] = 1
            det = preds[key]
            pred_recall, proposal = image_eval(det, boxes, keep, iou)
            curves[n] = image_pr(det[:, 4], proposal, pred_recall)
        out[level] = (curves, faces)
    return out


def ap_from_curves(curve: np.ndarray, faces: float) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = curve[:, 1] / curve[:, 0]
    return voc_ap(curve[:, 1] / faces, precision)


def evaluate(predictions: dict[str, np.ndarray], gt: dict[str, object], iou: float = 0.5):
    """AP for easy, medium and hard. Every ground-truth image must have a prediction entry."""
    curves = image_curves(predictions, gt, iou)
    return {
        level: round(ap_from_curves(c.sum(axis=0), f.sum()), 4) for level, (c, f) in curves.items()
    }


def bootstrap_weights(n_images: int, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """(n_boot, n_images) resampling counts; share them across detectors for paired differences."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_images, np.full(n_images, 1 / n_images), size=n_boot).astype(
        np.float64
    )


def bootstrap_ap(curves: np.ndarray, faces: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AP on each resample of images."""
    flat = curves.reshape(len(curves), -1)
    sums = (weights @ flat).reshape(len(weights), THRESH_NUM, 2)
    counts = weights @ faces
    return np.array([ap_from_curves(c, f) for c, f in zip(sums, counts, strict=True)])
