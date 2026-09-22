import numpy as np
import pytest

from faceid_bench.align import ARCFACE_112
from faceid_bench.failures import landmark_geometry, pair_stage, stage_of


def row(width=120.0, score=0.9, roll=0.0, yaw=0.0):
    r = np.zeros(15, np.float32)
    r[2] = r[3] = width
    r[14] = score
    points = ARCFACE_112.copy() * width / 112
    if roll:
        a = np.radians(roll)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        points = (points - points.mean(0)) @ rot.T + points.mean(0)
    if yaw:
        eye = points[1] - points[0]
        points[2] = points[2] + yaw * eye
    r[4:14] = points.reshape(-1)
    return r


def test_roll_and_yaw_read_from_landmarks():
    assert landmark_geometry(row())["roll_deg"] == pytest.approx(0, abs=0.5)
    assert landmark_geometry(row(roll=15))["roll_deg"] == pytest.approx(15, abs=0.5)
    assert landmark_geometry(row(yaw=0.4))["yaw_proxy"] == pytest.approx(
        landmark_geometry(row())["yaw_proxy"] + 0.4, abs=0.02
    )


def good(**kw):
    image = {
        "found": True,
        "face_width_px": 120.0,
        "det_score": 0.9,
        "roll_deg": 2.0,
        "yaw_proxy": 0.0,
        "eye_distance_px": 40.0,
        "sharpness": 400.0,
        "brightness": 120.0,
    }
    return {**image, **kw}


@pytest.mark.parametrize(
    "image,expected",
    [
        (good(), "embed: model"),
        (good(found=False), "detect: no face"),
        (good(face_width_px=50), "detect: small face"),
        (good(roll_deg=-35), "align: extreme pose"),
        (good(yaw_proxy=0.5), "align: extreme pose"),
        (good(sharpness=30), "image: blurred"),
        (good(brightness=30), "image: dark or washed out"),
        (good(brightness=230), "image: dark or washed out"),
    ],
)
def test_stage_rules(image, expected):
    assert stage_of(image, label_error=False) == expected


def test_label_error_wins_and_pair_takes_the_worse_image():
    assert stage_of(good(), label_error=True) == "data: wrong label"
    assert pair_stage(good(), good(roll_deg=40), False) == "align: extreme pose"
    assert pair_stage(good(found=False), good(sharpness=10), False) == "detect: no face"
