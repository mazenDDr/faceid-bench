"""Join verification accuracy (LFW) and embedder latency into results/verification.json + tables.

Inputs: outputs/verification/compare.json (pulled from gpu-box) and outputs/latency_*.jsonl.
"""

import json
from pathlib import Path

from faceid_bench.results import latency_rows, latency_summary, ms

# Published LFW 10-fold accuracy, for the reproduction check
PUBLISHED = {"mbf_w600k": 0.9970, "r50_w600k": 0.9983, "sface": 0.9940}
TIMED = {"mbf_w600k": "mbf_w600k_112", "r50_w600k": "r50_w600k_112", "sface": "sface_112"}
FP16_CHECKED = {"mbf_w600k", "r50_w600k"}  # sface fp16 export fails to load (BatchNorm dtype)

compare = json.loads(Path("outputs/verification/compare.json").read_text())
rows = latency_rows()
models = compare["models"]
out = {
    "test_people": compare["test_people"],
    "test_pairs": compare["test_pairs"],
    "baseline": compare["baseline"],
    "models": {},
}
for name, model in TIMED.items():
    row = models[name]
    fp16 = models.get(f"{name}_fp16")
    out["models"][name] = {
        "lfw_10fold": {**row["lfw_10fold"], "published": PUBLISHED[name]},
        "operating_points": row["operating_points"],
        "fp16_check": None
        if fp16 is None
        else {
            "lfw_10fold": fp16["lfw_10fold"]["accuracy"],
            "test_tar": {k: v["test_tar"] for k, v in fp16["operating_points"].items()},
        },
        "timed_model": model,
        "latency": latency_summary(rows, model, name in FP16_CHECKED),
    }
Path("results/verification.json").write_text(json.dumps(out, indent=2) + "\n")

print("| Model | LFW 10-fold (published) | TAR @ FAR 1e-3 | TAR @ 1e-4 | TAR @ 1e-5 |")
print("|---|---|---|---|---|")
for name, m in out["models"].items():
    cells = []
    for far in ("0.001", "0.0001", "1e-05"):
        o = m["operating_points"][far]
        lo, hi = o["test_tar_ci95"]
        cells.append(f"{o['test_tar']:.4f} [{lo:.4f}, {hi:.4f}]")
    lfw = m["lfw_10fold"]
    print(f"| {name} | {lfw['accuracy']:.4f} ({lfw['published']:.4f}) | {' | '.join(cells)} |")
print()
print("| Model | Test FAR at 1e-5 (false accepts) | Mac best | Mac CPU | RTX CUDA | RTX CPU |")
print("|---|---|---|---|---|---|")
for name, m in out["models"].items():
    o, lat = m["operating_points"]["1e-05"], m["latency"]
    print(
        f"| {name} | {o['test_far']:.1e} ({o['test_false_accepts']}) "
        f"| {ms(lat['mac-m4pro_best'], True)} | {ms(lat['mac-m4pro_cpu'])} "
        f"| {ms(lat['rtx5060ti_cuda'], True)} | {ms(lat['rtx5060ti_cpu'])} |"
    )
