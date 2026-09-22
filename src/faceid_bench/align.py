"""Five-point face alignment to the 112x112 ArcFace template.

Landmarks are in image order: left-of-image eye, right-of-image eye, nose, left mouth corner,
right mouth corner (both YuNet and SCRFD emit this order).
"""

from __future__ import annotations

import numpy as np

ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, shift) mapping src to dst.

    Umeyama (1991), as used by scikit-image's SimilarityTransform and InsightFace's norm_crop.
    Returns the 2x3 matrix for cv2.warpAffine.
    """
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    rotation = u @ np.diag(d) @ vt
    scale = (s * d).sum() / src_c.var(axis=0).sum()
    shift = dst_mean - scale * rotation @ src_mean
    return np.column_stack([scale * rotation, shift])


def align(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    import cv2

    matrix = umeyama(np.asarray(landmarks).reshape(5, 2), ARCFACE_112 * size / 112)
    return cv2.warpAffine(image, matrix, (size, size), borderValue=0)
