"""Collect the failure analysis into results/failures.json and print the stage tables.

Input: outputs/failures/analysis.json (written on the GPU machine).
"""

import json
from pathlib import Path

from faceid_bench.data import LFW_LABEL_ERRORS, LFW_NAME_COLLISIONS

analysis = json.loads(Path("outputs/failures/analysis.json").read_text())
out = {
    "far_target": analysis["far_target"],
    "threshold_from_dev": analysis["threshold_from_dev"],
    "test_pairs": analysis["test_pairs"],
    "known_data_problems": {"images": LFW_LABEL_ERRORS, "name_collisions": LFW_NAME_COLLISIONS},
}
for key in ("false_rejects", "false_accepts"):
    section = analysis[key]
    data_share = section["stages"].get("data: wrong label", 0)
    out[key] = {
        "count": section["count"],
        "stages": section["stages"],
        "model_only": section["count"] - data_share,
        "fixed_by_accurate_config": section["fixed_by_accurate_config"],
        "examples": section["examples"],
    }
Path("results/failures.json").write_text(json.dumps(out, indent=2) + "\n")

for key in ("false_rejects", "false_accepts"):
    section = out[key]
    kind = "rejected the right person" if key == "false_rejects" else "accepted the wrong person"
    print(
        f"\n**{section['count']} times the pipeline {kind}** (of "
        f"{analysis['test_pairs']['genuine' if key == 'false_rejects' else 'impostor']:,} pairs)"
    )
    print("| Stage that broke | Count | Share |")
    print("|---|---|---|")
    for stage, count in section["stages"].items():
        print(f"| {stage} | {count} | {count / section['count']:.0%} |")
    print(
        f"| **model's own failures** | **{section['model_only']}** | "
        f"{section['model_only'] / section['count']:.0%} |"
    )
    print(f"Recovered by the accurate configuration: {section['fixed_by_accurate_config']}")
