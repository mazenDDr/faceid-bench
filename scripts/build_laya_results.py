"""Collect the Laya decision-engine comparison into results/laya.json and print a table.

Inputs: outputs/laya/compare.json (gpu-box) and outputs/laya/latency_<machine>.json.
"""

import json
from pathlib import Path

compare = json.loads(Path("outputs/laya/compare.json").read_text())
files = [json.loads(f.read_text()) for f in sorted(Path("outputs/laya").glob("latency_*.json"))]
latency = {f["machine"]: f["devices"] for f in files}
platt_latency = {f["machine"]: f["platt_cosine"] for f in files if "platt_cosine" in f}
out = {**compare, "laya_latency": latency, "platt_latency": platt_latency}
Path("results/laya.json").write_text(json.dumps(out, indent=2) + "\n")

print(f"Test pairs: {compare['test_pairs'][0]} genuine, {compare['test_pairs'][1]} impostor")
print("| Decision method | AUC | TAR @ FAR 1e-3 | Cllr | ECE | Cllr vs Platt (paired) |")
print("|---|---|---|---|---|---|")
for name, m in compare["methods"].items():
    diff = m.get("cllr_minus_platt_cosine")
    d = (
        "—"
        if diff is None
        else f"{diff['mean']:+.3f} [{diff['ci95'][0]:+.3f}, {diff['ci95'][1]:+.3f}]"
    )
    lo, hi = m["tar_ci95"]
    clo, chi = m["cllr_ci95"]
    print(
        f"| {name} | {m['auc']:.4f} | {m['tar_at_far_1e-3']:.4f} [{lo:.4f}, {hi:.4f}] "
        f"| {m['cllr']:.4f} [{clo:.4f}, {chi:.4f}] | {m['ece']:.4f} | {d} |"
    )
print()
for machine, devices in latency.items():
    for device, v in devices.items():
        print(f"Laya on {machine} {device}: p50 {v['p50_ms']:.1f} ms per decision")
for machine, v in platt_latency.items():
    print(f"Platt on {machine}: {v['per_call_us']:.2f} microseconds per decision (Python + NumPy)")
