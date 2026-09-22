"""Collect calibration results (official labels + label-error sensitivity) into
results/calibration.json and print a table. Inputs: outputs/calibration/compare{,_clean}.json."""

import json
from pathlib import Path

from faceid_bench.data import LFW_LABEL_ERRORS

official = json.loads(Path("outputs/calibration/compare.json").read_text())
clean = json.loads(Path("outputs/calibration/compare_clean.json").read_text())
out = {
    "prior": official["prior"],
    "n_boot": official["n_boot"],
    "official_labels": official["models"],
    "label_errors_removed": {"images": LFW_LABEL_ERRORS, "models": clean["models"]},
}
Path("results/calibration.json").write_text(json.dumps(out, indent=2) + "\n")

print("| Model | Calibrator | Cllr (official labels) | ECE | Cllr (4 label errors removed) |")
print("|---|---|---|---|---|")
for name, row in official["models"].items():
    for cal, v in row["calibrators"].items():
        c = clean["models"][name]["calibrators"][cal]["cllr"]
        lo, hi = v["cllr"]["ci95"]
        clo, chi = c["ci95"]
        print(
            f"| {name} | {cal} | {v['cllr']['point']:.4f} [{lo:.4f}, {hi:.4f}] "
            f"| {v['ece']['point']:.4f} | {c['point']:.4f} [{clo:.4f}, {chi:.4f}] |"
        )
    print(
        f"| {name} | minimum possible | {row['min_cllr_test']:.4f} | — "
        f"| {clean['models'][name]['min_cllr_test']:.4f} |"
    )
