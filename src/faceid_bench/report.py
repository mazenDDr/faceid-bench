"""README pieces built from results/*.json: the hero SVG and the generated tables.

Nothing here holds a number of its own; every value is read from a results file, so the README
cannot drift from the runs.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

RESULTS = Path("results")

# categorical slots 1-4 of the dataviz reference palette, validated for adjacent stacks in both
# modes (light contrast of slots 3-4 is below 3:1, so every segment carries a visible label)
STAGES = [
    ("detect_ms", "find face", "#2a78d6", "#3987e5"),
    ("align_ms", "align", "#eb6834", "#d95926"),
    ("embed_ms", "embed", "#1baf7a", "#199e70"),
    ("decide_ms", "decide", "#eda100", "#c98500"),
]
MS_PER_SECOND_ON_SCREEN = 1000 / 60  # the animation plays 1 ms of compute as 60 ms
MACHINES = {"mac-m4pro": "Mac (Apple M4 Pro)", "rtx5060ti": "RTX 5060 Ti PC"}
DETECTOR_NAMES = {
    "scrfd_500m_kps": "SCRFD-500M",
    "scrfd_10g_kps": "SCRFD-10G",
    "yunet": "YuNet",
}


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def fmt_ms(v: float) -> str:
    return f"{v:.3f}" if v < 0.01 else f"{v:.2f}" if v < 10 else f"{v:.1f}"


def hero_svg(pipeline: dict) -> tuple[str, str]:
    """Two stacked timelines on one scale: the balanced pipeline, then the same with Laya."""
    rows = [
        ("Face ID-style pipeline", pipeline["configs"]["balanced"]["mac_coreml_all_fp16"]),
        ("Same, with Laya deciding", pipeline["configs"]["balanced+laya"]["mac_coreml_all_fp16"]),
    ]
    width, left, right = 760, 24, 24
    scale_max = max(r["total_ms"] for _, r in rows) * 1.02
    px = (width - left - right) / scale_max
    bar_h, row_gap, top = 34, 104, 146
    height = top + row_gap * (len(rows) - 1) + bar_h + 44

    parts, css = [], []
    # legend: identity never rests on colour alone
    lx = left
    for j, (_, name, _, _) in enumerate(STAGES):
        parts.append(f'<rect class="s{j}" x="{lx}" y="88" width="14" height="14" rx="3"/>')
        parts.append(f'<text x="{lx + 20}" y="100" class="legend">{html.escape(name)}</text>')
        lx += 20 + len(name) * 8 + 22
    for i, (label, times) in enumerate(rows):
        y = top + i * row_gap
        parts.append(
            f'<text x="{left}" y="{y - 12}" class="row">{html.escape(label)} '
            f'<tspan class="total">{fmt_ms(times["total_ms"])} ms</tspan></text>'
        )
        x, delay = float(left), 0.3 + i * 0.4
        for j, (key, name, _, _) in enumerate(STAGES):
            ms = times[key]
            w = ms * px
            duration = max(ms / MS_PER_SECOND_ON_SCREEN, 0.05)
            parts.append(
                f'<rect class="s{j} a{i}{j}" x="{x:.2f}" y="{y}" width="{max(w - 2, 0.6):.2f}" '
                f'height="{bar_h}" rx="4"/>'
            )
            css.append(
                f".a{i}{j}{{transform-origin:{x:.2f}px 0;animation:grow {duration:.2f}s linear "
                f"{delay:.2f}s both}}"
            )
            text = f"{name} {fmt_ms(ms)} ms"
            if w > len(text) * 8 + 16:  # only a segment wide enough gets a label inside
                parts.append(
                    f'<text x="{x + 10:.1f}" y="{y + 23}" class="seg">{html.escape(text)}</text>'
                )
            delay += duration
            x += w
        numbers = " · ".join(f"{name} {fmt_ms(times[key])}" for key, name, _, _ in STAGES)
        parts.append(
            f'<text x="{left}" y="{y + bar_h + 22}" class="note">{html.escape(numbers)} ms</text>'
        )

    light = "".join(f".s{j}{{fill:{c}}}" for j, (_, _, c, _) in enumerate(STAGES))
    dark = "".join(f".s{j}{{fill:{d}}}" for j, (_, _, _, d) in enumerate(STAGES))
    machine = MACHINES.get(pipeline["machine"], pipeline["machine"])
    style = f"""
svg{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.bg{{fill:#fcfcfb}} .title{{font-size:22px;font-weight:600;fill:#0b0b0b}}
.sub{{font-size:14px;fill:#52514e}} .row{{font-size:16px;font-weight:600;fill:#0b0b0b}}
.total{{font-weight:400;fill:#52514e}} .seg{{font-size:14px;fill:#ffffff;font-weight:600}}
.note,.legend{{font-size:14px;fill:#52514e}} {light}
@keyframes grow{{from{{transform:scaleX(0)}}to{{transform:scaleX(1)}}}}
{"".join(css)}
@media (prefers-color-scheme:dark){{.bg{{fill:#1a1a19}} .title,.row{{fill:#ffffff}}
.sub,.total,.note,.legend{{fill:#c3c2b7}} {dark}}}
@media (prefers-reduced-motion:reduce){{rect{{animation:none!important}}}}
"""
    title = "One unlock attempt, timed stage by stage"
    sub1 = f"{machine}, Core ML fp16, median of 3 sessions."
    sub2 = f"Bars grow at one real-time rate: 1 ms of compute = {1000 / MS_PER_SECOND_ON_SCREEN:.0f} ms on screen."
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="t d">'
        f'<title id="t">{html.escape(title)}</title>'
        f'<desc id="d">{html.escape(sub1 + " " + sub2)}</desc>'
        f"<style>{style}</style>"
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="12"/>'
        f'<text x="{left}" y="36" class="title">{html.escape(title)}</text>'
        f'<text x="{left}" y="58" class="sub">{html.escape(sub1)}</text>'
        f'<text x="{left}" y="76" class="sub">{html.escape(sub2)}</text>'
        + "".join(parts)
        + "</svg>\n"
    )
    b, lay = rows[0][1], rows[1][1]
    alt = (
        f"One unlock attempt on a {machine}: finding the face, aligning it, embedding it and "
        f"deciding takes {fmt_ms(b['total_ms'])} ms. With Laya making the decision it takes "
        f"{fmt_ms(lay['total_ms'])} ms, of which Laya is {fmt_ms(lay['decide_ms'])} ms."
    )
    return svg, alt


def replace_block(text: str, name: str, body: str) -> str:
    """Swap the text between <!-- BEGIN GENERATED name --> and its END marker."""
    pattern = re.compile(
        rf"(<!-- BEGIN GENERATED {re.escape(name)} -->\n).*?(<!-- END GENERATED {re.escape(name)} -->)",
        re.S,
    )
    if not pattern.search(text):
        raise KeyError(f"no GENERATED {name} block")
    return pattern.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text)


def ci(v: list[float], digits: int = 4) -> str:
    return f"[{v[0]:.{digits}f}, {v[1]:.{digits}f}]"


def pipeline_table(pipeline: dict) -> str:
    names = {"fast": "Fast", "balanced": "**Balanced**", "accurate": "Accurate"}
    lines = [
        "| Pipeline | Models | Accepts the right person, TAR @ FAR 1e-5 | Mac p50 / p95 per attempt |",
        "|---|---|---|---:|",
    ]
    for key, label in names.items():
        c = pipeline["configs"][key]
        cfg = c["config"]
        stem = Path(cfg["detector"]).stem.replace("_fp16", "")
        base, size = stem.rsplit("_", 1)
        det = f"{DETECTOR_NAMES[base]} @ {size} px"
        emb = "ResNet-50" if "r50" in cfg["embedder"] else "MobileFaceNet"
        op = c["operating_points"]["1e-05"]
        t = c["mac_coreml_all_fp16"]
        lines.append(
            f"| {label} | {det} + {emb} | {op['test_tar']:.4f} {ci(op['test_tar_ci95'])} "
            f"| {fmt_ms(t['total_ms'])} / {fmt_ms(t['total_p95_ms'])} ms |"
        )
    laya = pipeline["configs"]["balanced+laya"]
    t, a = laya["mac_coreml_all_fp16"], laya["decision_accuracy"]
    lines.append(
        f"| Balanced, **Laya decides** | same, Laya zero-shot on the Mac GPU | {a['tar_at_far_1e-3']:.4f} "
        f"@ FAR 1e-3 | {fmt_ms(t['total_ms'])} / {fmt_ms(t['total_p95_ms'])} ms |"
    )
    ver = load("verification")
    pairs = ver["test_pairs"]
    lines += [
        "",
        f"*Accuracy: LFW, {ver['test_people']:,} people never seen while tuning. The threshold is "
        f"fixed on separate dev people for 1 false accept in 100,000, then applied to "
        f"{pairs['genuine']:,} same-person and {pairs['impostor']:,} different-person pairs. Time: one "
        f"attempt on a {pipeline['frames'].split(',')[0]} frame on a {MACHINES[pipeline['machine']]}, "
        f"Core ML fp16, median of 3 sessions. The Laya row is measured at a looser setting "
        f"(1 in 1,000).*",
    ]
    return "\n".join(lines)


def laya_table(laya: dict) -> str:
    rows = [
        ("Platt on the cosine (2 numbers)", "platt_cosine", "platt"),
        ("Laya zero-shot, number only", "laya_plain", "laya"),
        ("Laya zero-shot, number + one sentence", "laya_note", "laya"),
        ("Laya zero-shot, rich JSON", "laya_rich", "laya"),
        ("**Laya fine-tuned** on dev people", "laya_note_finetuned", "laya"),
    ]
    mac_laya = laya["laya_latency"]["mac-m4pro"]["mps"]["p50_ms"]
    mac_platt = laya["platt_latency"]["mac-m4pro"]["per_call_us"]
    lines = [
        "| Decision step | AUC | TAR @ FAR 1e-3 | Cllr (lower is better) | vs Platt, paired | Mac time per decision |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for label, key, kind in rows:
        m = laya["methods"][key]
        diff = m.get("cllr_minus_platt_cosine")
        verdict = (
            "—"
            if diff is None
            else (
                f"{diff['mean']:+.3f} {ci(diff['ci95'], 3)}"
                + ("" if diff["significant"] else " (tie)")
            )
        )
        cost = f"{mac_platt:.2f} µs" if kind == "platt" else f"{mac_laya:.1f} ms"
        if key.endswith("finetuned"):
            cost += "¹"
        lines.append(
            f"| {label} | {m['auc']:.4f} | {m['tar_at_far_1e-3']:.4f} | {m['cllr']:.4f} | {verdict} | {cost} |"
        )
    t = laya["test_pairs"]
    lines += [
        "",
        f"*{t[0]:,} same-person and {t[1]:,} different-person pairs of test people; 95% intervals "
        f"resample people. Time: one decision on the Mac GPU (Laya) or CPU (Platt), median of 3 "
        f"sessions. ¹ Timed with the base model; the fine-tuned model has the same architecture.*",
    ]
    return "\n".join(lines)


def detector_table(det: dict) -> str:
    names = {
        "yunet_2026may": "YuNet 640",
        "scrfd_500m_kps": "SCRFD-500M 640",
        "scrfd_10g_kps": "SCRFD-10G 640",
        "scrfd_500m_kps@320": "SCRFD-500M 320",
        "scrfd_10g_kps@320": "SCRFD-10G 320",
        "yunet@320": "YuNet 320",
    }
    lines = [
        "| Detector | Easy AP | Medium AP | Hard AP | Mac, model only |",
        "|---|---|---|---|---:|",
    ]
    for key, label in names.items():
        d = det["detectors"][key]
        ap = d["wider_val"]
        cells = " | ".join(
            f"{ap[k]['ap']:.3f} {ci(ap[k]['ci95'], 3)}" for k in ("easy", "medium", "hard")
        )
        best = d["latency"]["mac-m4pro_best"]
        lines.append(f"| {label} | {cells} | {fmt_ms(best['p50_ms'])} ms |")
    return "\n".join(lines)


def verification_table(ver: dict) -> str:
    names = {
        "r50_w600k": "ResNet-50 (WebFace600K)",
        "mbf_w600k": "MobileFaceNet (WebFace600K)",
        "sface": "SFace",
    }
    lines = [
        "| Face model | LFW 10-fold (published) | TAR @ 1e-3 | TAR @ 1e-4 | TAR @ 1e-5 | Mac, model only |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, label in names.items():
        m = ver["models"][key]
        lfw = m["lfw_10fold"]
        ops = " | ".join(
            f"{m['operating_points'][f]['test_tar']:.4f}" for f in ("0.001", "0.0001", "1e-05")
        )
        best = m["latency"]["mac-m4pro_best"]
        lines.append(
            f"| {label} | {lfw['accuracy']:.4f} ({lfw['published']:.4f}) | {ops} | {fmt_ms(best['p50_ms'])} ms |"
        )
    return "\n".join(lines)


def calibration_table(cal: dict) -> str:
    r = cal["official_labels"]["r50_w600k"]
    names = {
        "naive": "Read the cosine as a probability",
        "platt": "**Platt** (2 numbers)",
        "isotonic": "Isotonic",
    }
    lines = ["| Score → probability | Cllr | ECE |", "|---|---|---:|"]
    for key, label in names.items():
        c = r["calibrators"][key]
        lines.append(
            f"| {label} | {c['cllr']['point']:.4f} {ci(c['cllr']['ci95'])} | {c['ece']['point']:.4f} |"
        )
    lines.append(f"| Best possible for this model | {r['min_cllr_test']:.4f} | — |")
    return "\n".join(lines)


def failure_table(fail: dict) -> str:
    fr, fa = fail["false_rejects"], fail["false_accepts"]
    stages = list(dict.fromkeys([*fr["stages"], *fa["stages"]]))
    lines = [
        f"| Stage that broke | Rejected the right person ({fr['count']}) | Accepted the wrong person ({fa['count']}) |",
        "|---|---:|---:|",
    ]
    for s in stages:
        lines.append(f"| {s} | {fr['stages'].get(s, 0)} | {fa['stages'].get(s, 0)} |")
    lines.append(
        f"| fixed by the 640 px detector | {fr['fixed_by_accurate_config']} | {fa['fixed_by_accurate_config']} |"
    )
    return "\n".join(lines)
