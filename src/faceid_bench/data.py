"""Readers for WIDER FACE (detection) and LFW (verification), and the identity-level split.

Verification is tuned (threshold, calibration) on dev identities and reported on test identities.
No person appears on both sides, so a model cannot look better by having seen a face before.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path("data")


# --- WIDER FACE -------------------------------------------------------------------------------


@dataclass
class WiderImage:
    path: str  # relative to WIDER_val/images, e.g. "0--Parade/0_Parade_marchingband_1_465.jpg"
    boxes: np.ndarray  # (n, 4) x, y, w, h
    invalid: np.ndarray  # (n,) bool, the annotators' "invalid" flag


def read_wider_gt(path: str | Path) -> list[WiderImage]:
    """Parse wider_face_{split}_bbx_gt.txt.

    Each record is: image path, face count, then one line per face with
    `x y w h blur expression illumination invalid occlusion pose`. An image with zero faces
    still has one all-zero line.
    """
    lines = Path(path).read_text().split("\n")
    images, i = [], 0
    while i < len(lines) and lines[i].strip():
        name, count = lines[i].strip(), int(lines[i + 1])
        rows = [lines[i + 2 + k].split() for k in range(max(count, 1))]
        values = np.array(rows, dtype=np.int64).reshape(-1, 10)[:count]
        images.append(WiderImage(name, values[:, :4].astype(np.float32), values[:, 7] == 1))
        i += 2 + max(count, 1)
    return images


def read_wider_subsets(eval_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Official easy/medium/hard subsets: for each image, the 1-based indices of faces that count.

    Read from eval_tools/ground_truth/wider_{easy,medium,hard}_val.mat, the files the official
    MATLAB evaluation uses. Faces not listed are ignored (neither hits nor misses).
    """
    from scipy.io import loadmat

    subsets = {}
    for level in ("easy", "medium", "hard"):
        mat = loadmat(Path(eval_dir) / f"wider_{level}_val.mat")
        keep = {}
        for event_files, event_keep in zip(
            mat["file_list"][:, 0], mat["gt_list"][:, 0], strict=True
        ):
            for name, idx in zip(event_files[:, 0], event_keep[:, 0], strict=True):
                keep[str(name[0])] = idx.reshape(-1).astype(np.int64)
        subsets[level] = keep
    return subsets


# --- LFW --------------------------------------------------------------------------------------


def lfw_images(root: str | Path = DATA / "lfw" / "lfw") -> dict[str, list[str]]:
    """identity -> sorted image paths relative to `root`."""
    root = Path(root)
    return {
        person.name: sorted(str(p.relative_to(root)) for p in person.glob("*.jpg"))
        for person in sorted(root.iterdir())
        if person.is_dir()
    }


def lfw_image_path(name: str, number: int) -> str:
    return f"{name}/{name}_{number:04d}.jpg"


def read_lfw_pairs(path: str | Path) -> list[tuple[int, str, str, bool]]:
    """Official pairs.txt (10 folds x 300 same + 300 different) as (fold, img_a, img_b, same).

    The first line holds `folds pairs_per_class`; files with a single number have one fold.
    """
    lines = [line.split() for line in Path(path).read_text().strip().split("\n")]
    header, rows = lines[0], lines[1:]
    folds, per_class = (int(header[0]), int(header[1])) if len(header) == 2 else (1, int(header[0]))
    pairs = []
    for k, row in enumerate(rows):
        fold = k // (2 * per_class)
        if len(row) == 3:
            pairs.append(
                (
                    fold,
                    lfw_image_path(row[0], int(row[1])),
                    lfw_image_path(row[0], int(row[2])),
                    True,
                )
            )
        else:
            pairs.append(
                (
                    fold,
                    lfw_image_path(row[0], int(row[1])),
                    lfw_image_path(row[2], int(row[3])),
                    False,
                )
            )
    assert fold == folds - 1, f"expected {folds} folds, parsed {fold + 1}"
    return pairs


def split_of(identity: str, test_share: float = 0.5, salt: str = "faceid-bench") -> str:
    """'dev' or 'test' from a hash of the identity name: stable, no randomness to record."""
    digest = hashlib.sha256(f"{salt}:{identity}".encode()).digest()
    return "test" if int.from_bytes(digest[:8], "big") / 2**64 < test_share else "dev"


def identity_split(
    images: dict[str, list[str]], max_per_identity: int = 20
) -> dict[str, dict[str, list[str]]]:
    """Split identities into dev/test and cap images per identity.

    The cap stops a few heavily photographed people (LFW has one with 530 images) from
    dominating the genuine pairs. The first `max_per_identity` images by file order are kept.
    """
    split: dict[str, dict[str, list[str]]] = {"dev": {}, "test": {}}
    for identity, paths in images.items():
        split[split_of(identity)][identity] = paths[:max_per_identity]
    return split


def pair_counts(identities: dict[str, list[str]]) -> dict[str, int]:
    """Number of genuine (same person) and impostor (different people) pairs, all-vs-all."""
    sizes = np.array([len(p) for p in identities.values()], dtype=np.int64)
    n = int(sizes.sum())
    genuine = int((sizes * (sizes - 1) // 2).sum())
    return {
        "identities": len(sizes),
        "images": n,
        "identities_with_2plus": int((sizes >= 2).sum()),
        "genuine_pairs": genuine,
        "impostor_pairs": n * (n - 1) // 2 - genuine,
    }
