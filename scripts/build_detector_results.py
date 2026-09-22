"""Join detector accuracy (with intervals) and latency into results/detectors.json + a table.

Inputs (pulled from gpu-box / written on the Mac): outputs/wider/compare.json and
outputs/latency_<machine>.jsonl. Latency is the model forward pass only (batch 1); pre- and
post-processing are timed with the full pipeline later.
"""

import json
from pathlib import Path

from faceid_bench.results import latency_rows, latency_summary, ms

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
rows = latency_rows()


out = {"baseline": compare["baseline"], "n_boot": compare["n_boot"], "detectors": {}}
for name, model in TIMED.items():
    fp16_ok = name.split("@")[0] in FP16_CHECKED
    out["detectors"][name] = {
        "wider_val": compare["detectors"][name],
        "timed_model": model,
        "latency": latency_summary(rows, model, fp16_ok),
    }
Path("results/detectors.json").write_text(json.dumps(out, indent=2) + "\n")


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
