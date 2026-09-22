"""Per-image quality features and the rule that assigns a failure to the stage that broke."""

from __future__ import annotations

import numpy as np

# Thresholds are percentiles of the dev-people distribution (800 images, measured with
# `scripts/failure_analysis.py --dev-quantiles`), never chosen from the test failures.
# "Unusual for this dataset" means: p5 for size and darkness, p10 for blur, p99 for pose
# and brightness.
SMALL_FACE_PX = 82.0  # dev p5 82.6
HIGH_ROLL_DEG = 15.0  # dev p99 14.6
HIGH_YAW = 0.45  # dev p99 0.47
LOW_SHARPNESS = 81.0  # dev p10 81.1
DARK = 79.0  # dev p5 78.9
BRIGHT = 163.0  # dev p99 163.0


def landmark_geometry(row: np.ndarray) -> dict[str, float]:
    """Roll and a yaw proxy from the five landmarks of one detection row."""
    points = row[4:14].reshape(5, 2)
    right_eye, left_eye, nose = points[0], points[1], points[2]
    eye_vector = left_eye - right_eye
    eye_distance = float(np.linalg.norm(eye_vector)) or 1.0
    middle = (right_eye + left_eye) / 2
    # nose offset along the eye line, in eye-distance units: 0 is frontal, +-0.5 is a profile
    along = float((nose - middle) @ eye_vector) / eye_distance**2
    return {
        "roll_deg": float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))),
        "yaw_proxy": along,
        "eye_distance_px": eye_distance,
    }


def image_quality(crop: np.ndarray) -> dict[str, float]:
    """Sharpness (variance of the Laplacian) and brightness of an aligned crop."""
    import cv2

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        "brightness": float(grey.mean()),
    }


def features(row: np.ndarray, found: bool, crop: np.ndarray) -> dict[str, float]:
    return {
        "found": bool(found),
        "face_width_px": float(row[2]),
        "det_score": float(row[14]),
        **landmark_geometry(row),
        **image_quality(crop),
    }


def stage_of(image: dict, label_error: bool) -> str:
    """Which stage broke for this image, taking the first rule that fires.

    Order matters: a wrong label makes everything after it meaningless, a face that was never
    found cannot be aligned, and a badly posed crop explains a weak embedding.
    """
    if label_error:
        return "data: wrong label"
    if not image["found"]:
        return "detect: no face"
    if image["face_width_px"] < SMALL_FACE_PX:
        return "detect: small face"
    if abs(image["roll_deg"]) > HIGH_ROLL_DEG or abs(image["yaw_proxy"]) > HIGH_YAW:
        return "align: extreme pose"
    if image["sharpness"] < LOW_SHARPNESS:
        return "image: blurred"
    if image["brightness"] < DARK or image["brightness"] > BRIGHT:
        return "image: dark or washed out"
    return "embed: model"


def pair_stage(a: dict, b: dict, label_error: bool) -> str:
    """The worse of the two images decides the stage; ties keep the first."""
    order = [
        "data: wrong label",
        "detect: no face",
        "detect: small face",
        "align: extreme pose",
        "image: blurred",
        "image: dark or washed out",
        "embed: model",
    ]
    stages = [stage_of(a, label_error), stage_of(b, label_error)]
    return min(stages, key=order.index)
