import numpy as np
import pytest

from faceid_bench.data import (
    identity_split,
    pair_counts,
    read_lfw_pairs,
    read_wider_gt,
    split_of,
)

WIDER_GT = """0--Parade/a.jpg
2
10 20 30 40 0 0 0 0 0 0
50 60 5 6 2 0 0 1 0 0
0--Parade/empty.jpg
0
0 0 0 0 0 0 0 0 0 0
1--Handshaking/b.jpg
1
1 2 3 4 0 0 0 0 0 0
"""


def test_read_wider_gt_handles_zero_face_images(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(WIDER_GT)
    images = read_wider_gt(path)
    assert [im.path for im in images] == [
        "0--Parade/a.jpg",
        "0--Parade/empty.jpg",
        "1--Handshaking/b.jpg",
    ]
    assert images[0].boxes.tolist() == [[10, 20, 30, 40], [50, 60, 5, 6]]
    assert images[0].invalid.tolist() == [False, True]
    assert images[1].boxes.shape == (0, 4)
    assert images[2].boxes.tolist() == [[1, 2, 3, 4]]


def test_read_lfw_pairs_folds_and_labels(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("2\t1\nAnn\t1\t2\nBob\t1\tCy\t3\nDee\t2\t4\nEve\t1\tFay\t1\n")
    pairs = read_lfw_pairs(path)
    assert pairs[0] == (0, "Ann/Ann_0001.jpg", "Ann/Ann_0002.jpg", True)
    assert pairs[1] == (0, "Bob/Bob_0001.jpg", "Cy/Cy_0003.jpg", False)
    assert [p[0] for p in pairs] == [0, 0, 1, 1]


def test_split_is_stable_and_identity_disjoint():
    names = [f"person_{i}" for i in range(2000)]
    first = {n: split_of(n) for n in names}
    assert first == {n: split_of(n) for n in names}
    share = sum(v == "test" for v in first.values()) / len(names)
    assert 0.45 < share < 0.55
    split = identity_split({n: [f"{n}/{j}.jpg" for j in range(3)] for n in names})
    assert not set(split["dev"]) & set(split["test"])


def test_identity_split_caps_images():
    split = identity_split({"Big": [f"Big/{j}.jpg" for j in range(530)]}, max_per_identity=20)
    side = "test" if "Big" in split["test"] else "dev"
    assert len(split[side]["Big"]) == 20


def test_pair_counts():
    counts = pair_counts({"a": ["1", "2", "3"], "b": ["4"], "c": ["5", "6"]})
    assert counts["images"] == 6
    assert counts["genuine_pairs"] == 3 + 0 + 1
    assert counts["impostor_pairs"] == 15 - 4
    assert counts["identities_with_2plus"] == 2


def test_real_wider_subsets_if_downloaded():
    from pathlib import Path

    from faceid_bench.data import read_wider_subsets

    tools = Path("data/wider_face/eval_tools/ground_truth")
    if not tools.exists():
        pytest.skip("WIDER FACE eval tools not downloaded on this machine")
    subsets = read_wider_subsets(tools)
    assert len(subsets["hard"]) == 3226
    assert np.all([len(v) >= 0 for v in subsets["easy"].values()])
