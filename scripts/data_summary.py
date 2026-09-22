"""Count what was downloaded and check the official files agree. Run on gpu-box after download."""

import json
from pathlib import Path

import numpy as np

from faceid_bench.data import (
    DATA,
    identity_split,
    lfw_images,
    pair_counts,
    read_lfw_pairs,
    read_wider_gt,
    read_wider_subsets,
)

wider_root = DATA / "wider_face"
gt = read_wider_gt(wider_root / "wider_face_split" / "wider_face_val_bbx_gt.txt")
subsets = read_wider_subsets(wider_root / "eval_tools" / "ground_truth")
missing_images = [
    im.path for im in gt if not (wider_root / "WIDER_val" / "images" / im.path).exists()
]
by_name = {Path(im.path).stem: im for im in gt}
wider = {
    "images": len(gt),
    "faces": int(sum(len(im.boxes) for im in gt)),
    "faces_invalid": int(sum(im.invalid.sum() for im in gt)),
    "images_missing_on_disk": len(missing_images),
    "subset_faces": {
        level: int(sum(len(v) for v in keep.values())) for level, keep in subsets.items()
    },
    "subset_images_match_gt": all(set(keep) == set(by_name) for keep in subsets.values()),
    "subset_index_in_range": all(
        int(idx.max(initial=0)) <= len(by_name[name].boxes)
        for keep in subsets.values()
        for name, idx in keep.items()
    ),
}

images = lfw_images()
all_paths = {p for paths in images.values() for p in paths}
official = read_lfw_pairs(DATA / "lfw" / "pairs.txt")
split = identity_split(images)
sizes = np.array([len(v) for v in images.values()])
lfw = {
    "identities": len(images),
    "images": int(sizes.sum()),
    "max_images_per_identity": int(sizes.max()),
    "official_pairs": len(official),
    "official_pairs_same": sum(p[3] for p in official),
    "official_pair_images_missing": sum(
        a not in all_paths or b not in all_paths for _, a, b, _ in official
    ),
    "split": {side: pair_counts(ids) for side, ids in split.items()},
    "split_identity_overlap": len(set(split["dev"]) & set(split["test"])),
}

summary = {"wider_face_val": wider, "lfw": lfw}
out = Path("outputs/data_summary.json")
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
