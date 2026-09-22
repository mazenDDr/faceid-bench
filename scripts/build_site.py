"""Build site/index.html from site/index-template.html and results/*.json."""

import json
from pathlib import Path

from faceid_bench import report

pipeline = report.load("pipeline")
laya = report.load("laya")
balanced = pipeline["configs"]["balanced"]
laya_row = pipeline["configs"]["balanced+laya"]
machine = report.MACHINES[pipeline["machine"]]
ops = balanced["operating_points"]

rows = [
    ("Two numbers (Platt)", "platt_cosine"),
    ("Laya zero-shot, number only", "laya_plain"),
    ("Laya zero-shot, number + one sentence", "laya_note"),
    ("Laya zero-shot, rich JSON", "laya_rich"),
    ("Laya fine-tuned on dev people", "laya_note_finetuned"),
]
mac_laya = laya["laya_latency"]["mac-m4pro"]["mps"]["p50_ms"]
mac_platt = laya["platt_latency"]["mac-m4pro"]["per_call_us"]
laya_rows = []
for label, key in rows:
    m = laya["methods"][key]
    diff = m.get("cllr_minus_platt_cosine")
    versus = (
        "—"
        if diff is None
        else (
            f"Cllr {diff['mean']:+.3f} [{diff['ci95'][0]:+.3f}, {diff['ci95'][1]:+.3f}]"
            + (" — a tie" if not diff["significant"] else "")
        )
    )
    laya_rows.append(
        {
            "label": label,
            "auc": m["auc"],
            "tar": m["tar_at_far_1e-3"],
            "cllr": m["cllr"],
            "versus": versus,
            "time": f"{mac_platt:.2f} µs (Mac CPU)"
            if key == "platt_cosine"
            else f"{mac_laya:.1f} ms (Mac GPU)",
        }
    )

data = {
    "lede": (
        f"A Face ID-style unlock (find face, align, embed, decide) runs in "
        f"{report.fmt_ms(balanced['mac_coreml_all_fp16']['total_ms'])} ms on a {machine}. Where "
        f"to draw the unlock line is a trade-off between letting you in and keeping strangers out; "
        f"move it below and watch both change on real test pairs."
    ),
    "scores": report.load("scores"),
    "platt": balanced["config"]["platt"],
    "presets": [
        {
            "label": f"1 in {round(1 / float(far)):,} (dev)",
            "threshold": ops[far]["threshold_from_dev"],
        }
        for far in ("0.001", "0.0001", "1e-05")
    ],
    "timeline": [
        {"label": "Face ID-style pipeline", "times": balanced["mac_coreml_all_fp16"]},
        {"label": "Same, with Laya deciding", "times": laya_row["mac_coreml_all_fp16"]},
    ],
    "timeline_note": (
        f"{machine}, Core ML fp16, median of 3 sessions. Hover a segment for its time. "
        f"The bars grow at one real-time rate: 1 ms of compute = 60 ms on screen."
    ),
    "laya": laya_rows,
}
payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
template = Path("site/index-template.html").read_text()
assert template.count("/*__DATA__*/null") == 1
Path("site/index.html").write_text(template.replace("/*__DATA__*/null", payload))
print(f"site/index.html: {len(payload):,} bytes of data")
