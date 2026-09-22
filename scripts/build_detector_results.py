"""Join detector accuracy (with intervals) and latency into results/detectors.json + a table.

Inputs (pulled from gpu-box / written on the Mac): outputs/wider/compare.json and
outputs/latency_<machine>.jsonl. Latency is the model forward pass only (batch 1); pre- and
post-processing are timed with the full pipeline later.
"""

import json
from pathlib import Path

# WIDER detector name -> fixed-shape model used for timing
TIMED = {
    "yunet_2026may": "yunet_640",
    "yunet@320": "yunet_320",
    "scrfd_500m_kps": "scrfd_500m_kps_640",
    "scrfd_500m_kps@320": "scrfd_500m_kps_320",
    "scrfd_10g_kps": "scrfd_10g_kps_640",
    "scrfd_10g_kps@320": "scrfd_10g_kps_320",
}
# fp16 timings count only where fp16 AP was measured and matched fp32
FP16_CHECKED = {"scrfd_500m_kps", "scrfd_10g_kps"}

compare = json.loads(Path("outputs/wider/compare.json").read_text())
rows = []
for f in sorted(Path("outputs").glob("latency_*.jsonl")):
    rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
rows = [r for r in rows if not r["note"]]  # drop any silent fallback


def best(model: str, machine: str, fp16_ok: bool, backends: set[str]) -> dict | None:
    names = {model} | ({f"{model}_fp16"} if fp16_ok else set())
    found = [
        r
        for r in rows
        if r["model"] in names and r["machine"] == machine and r["backend"] in backends
    ]
    if not found:
        return None
    r = min(found, key=lambda r: r["p50_ms"])
    precision = "fp16" if r["model"].endswith("_fp16") else "fp32"
    return {
        "p50_ms": r["p50_ms"],
        "p95_ms": r["p95_ms"],
        "setting": f"{r['backend']}:{r['unit']}:{precision}",
        "session_p50_ms": r["session_p50_ms"],
    }


out = {"baseline": compare["baseline"], "n_boot": compare["n_boot"], "detectors": {}}
for name, model in TIMED.items():
    fp16_ok = name.split("@")[0] in FP16_CHECKED
    out["detectors"][name] = {
        "wider_val": compare["detectors"][name],
        "timed_model": model,
        "latency": {
            "mac-m4pro_best": best(model, "mac-m4pro", fp16_ok, {"ort-coreml", "ort-cpu"}),
            "mac-m4pro_cpu": best(model, "mac-m4pro", False, {"ort-cpu"}),
            "rtx5060ti_cuda": best(model, "rtx5060ti", fp16_ok, {"ort-cuda"}),
            "rtx5060ti_cpu": best(model, "rtx5060ti", False, {"ort-cpu"}),
        },
    }
Path("results/detectors.json").write_text(json.dumps(out, indent=2) + "\n")


def ms(v: dict | None, show_setting: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v['p50_ms']:.2f}" + (f" ({v['setting'].split(':', 1)[1]})" if show_setting else "")


print("| Detector | Easy AP | Medium AP | Hard AP | Mac best | Mac CPU | RTX CUDA | RTX host CPU |")
print("|---|---|---|---|---|---|---|---|")
for name, d in out["detectors"].items():
    ap = d["wider_val"]
    cells = []
    for level in ("easy", "medium", "hard"):
        lo, hi = ap[level]["ci95"]
        cells.append(f"{ap[level]['ap']:.3f} [{lo:.3f}, {hi:.3f}]")
    lat = d["latency"]
    print(
        f"| {name} | {' | '.join(cells)} "
        f"| {ms(lat['mac-m4pro_best'], True)} | {ms(lat['mac-m4pro_cpu'])} "
        f"| {ms(lat['rtx5060ti_cuda'], True)} | {ms(lat['rtx5060ti_cpu'])} |"
    )
