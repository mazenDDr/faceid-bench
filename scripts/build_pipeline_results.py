"""Join the Mac pipeline benchmark with each configuration's LFW accuracy and calibration
into results/pipeline.json, and print a table.

Inputs: outputs/pipeline/bench_mac-m4pro.json, outputs/verification/compare_pipeline.json,
outputs/calibration/compare_pipeline.json.
"""

import json
from pathlib import Path

bench = json.loads(Path("outputs/pipeline/bench_mac-m4pro.json").read_text())
verification = json.loads(Path("outputs/verification/compare_pipeline.json").read_text())
calibration = json.loads(Path("outputs/calibration/compare_pipeline.json").read_text())
# with Laya deciding, accuracy is Laya's own (note variant, recalibrated), measured at FAR 1e-3
laya = json.loads(Path("results/laya.json").read_text())["methods"]["laya_note+platt"]
ACCURACY_KEY = {
    "fast": "mbf_w600k__scrfd_500m_kps@320",
    "balanced": "r50_w600k__scrfd_10g_kps@320",
    "accurate": "r50_w600k",
    "balanced+laya": "r50_w600k__scrfd_10g_kps@320",
}
out = {"machine": bench["machine"], "frames": bench["frames"], "configs": {}}
for name, key in ACCURACY_KEY.items():
    b = bench[name]
    v = verification["models"][key]
    c = calibration["models"][key]["calibrators"]["platt"]
    entry = {
        "config": b["config"],
        "mac_coreml_all_fp16": b["coreml_all_fp16"],
        "session_total_p50_ms": b["session_total_p50_ms"],
        "fidelity_vs_cpu_fp32": b.get("fidelity_vs_cpu_fp32"),
    }
    if name.endswith("laya"):
        entry["decision_accuracy"] = {
            "source": "results/laya.json, laya_note+platt",
            "tar_at_far_1e-3": laya["tar_at_far_1e-3"],
            "tar_ci95": laya["tar_ci95"],
            "cllr": laya["cllr"],
        }
    else:
        entry.update(
            lfw_10fold=v["lfw_10fold"], operating_points=v["operating_points"], platt_cllr=c["cllr"]
        )
    out["configs"][name] = entry
Path("results/pipeline.json").write_text(json.dumps(out, indent=2) + "\n")

print("| Config | Accuracy (TAR) | Mac total p50 / p95 | detect | align | embed | decide |")
print("|---|---|---|---|---|---|---|")
for name, r in out["configs"].items():
    t = r["mac_coreml_all_fp16"]
    if "decision_accuracy" in r:
        a = r["decision_accuracy"]
        acc = f"{a['tar_at_far_1e-3']:.4f} @ FAR 1e-3 (Laya decides)"
    else:
        op = r["operating_points"]["1e-05"]
        lo, hi = op["test_tar_ci95"]
        acc = f"{op['test_tar']:.4f} [{lo:.4f}, {hi:.4f}] @ FAR 1e-5"
    print(
        f"| {name} | {acc} "
        f"| {t['total_ms']:.2f} / {t['total_p95_ms']:.2f} ms | {t['detect_ms']:.2f} "
        f"| {t['align_ms']:.2f} | {t['embed_ms']:.2f} | {t['decide_ms']:.3f} |"
    )
