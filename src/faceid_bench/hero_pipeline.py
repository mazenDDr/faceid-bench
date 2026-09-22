"""The "how one unlock attempt works" animation, drawn from results/trace.json.

Two real attempts play in a loop: the enrolled person (unlocks), then a stranger (stays
locked). Each goes through the four stages with its real numbers: the detection box, the five
landmarks flying to the alignment template, the cosine built from 32 real contributions, and
the Platt curve carrying the score to a probability. Pure CSS keyframes, so it animates inside
GitHub's <img>; with reduced motion it rests on the finished unlock.
"""

from __future__ import annotations

import html
import math

T = 18.0  # seconds per loop: genuine phase, stranger phase, hold
PHASES = {"genuine": 0.0, "impostor": 50.0}  # start of each phase, % of the loop
W, H = 960, 480
# panels
FX, FY, FS = 31, 132, 0.40  # frame panel origin and scale (640x480 -> 256x192)
CX, CY, CS = 334, 136, 1.25  # alignment crop origin and scale (112 -> 140)
BX0, BY, BSCALE = 522, 232, 1250.0  # contribution bars: left x, baseline y, px per unit
DX0, DX1, DY0, DY1 = 780, 930, 300, 136  # decide plot box (cosine -0.2..1.0, P 0..1)
COS0, COS1 = -0.2, 1.0

GREEN, RED = "#1a7f37", "#c4312f"
GREEN_D, RED_D = "#4ac26b", "#ff7b72"


def fmt(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def key(name: str, frames: list[tuple[float, str]]) -> str:
    body = "".join(f"{p:.2f}%{{{css}}}" for p, css in frames)
    return f"@keyframes {name}{{{body}}}"


def pipeline_svg(trace: dict) -> tuple[str, str]:
    a, b = trace["platt"]
    thr = trace["threshold"]
    times = trace["times_ms"]
    css, parts = [], []

    def dx(c: float) -> float:
        return DX0 + (c - COS0) / (COS1 - COS0) * (DX1 - DX0)

    def dy(p: float) -> float:
        return DY0 + p * (DY1 - DY0)

    # ---------------------------------------------------------------- static scaffolding
    parts.append(f'<rect class="bg" width="{W}" height="{H}" rx="14"/>')
    parts.append('<text x="24" y="40" class="title">How one unlock attempt works</text>')
    parts.append(
        '<text x="24" y="64" class="sub">A real LFW pair through the balanced pipeline: '
        "the enrolled person, then a stranger. Photos not shown; every number is real.</text>"
    )
    heads = [
        (24, f"1 · find face  {fmt(times['detect_ms'])} ms", "c0"),
        (318, f"2 · align  {fmt(times['align_ms'])} ms", "c1"),
        (512, f"3 · embed + compare  {fmt(times['embed_ms'])} ms", "c2"),
        (766, f"4 · decide  {times['decide_ms']:.3f} ms", "c3"),
    ]
    # stage windows in % of one phase: the underline under a stage lights up while it runs
    windows = [(1.5, 9), (8, 20.5), (20.5, 32.5), (32, 44)]
    for k, ((x, text, cls), (w0, w1)) in enumerate(zip(heads, windows, strict=True)):
        parts.append(f'<rect x="{x}" y="88" width="8" height="8" rx="2" class="{cls}f"/>')
        parts.append(f'<text x="{x + 14}" y="97" class="head">{html.escape(text)}</text>')
        frames = [(0, "opacity:0")]
        for start in PHASES.values():
            frames += [
                (start + w0, "opacity:0"),
                (start + w0 + 1, "opacity:1"),
                (start + w1 - 1, "opacity:1"),
                (start + w1, "opacity:0"),
            ]
        frames.append((100, "opacity:0"))
        css.append(key(f"u{k}", frames))
        css.append(f".u{k}{{animation:u{k} {T}s linear infinite}}")
        parts.append(
            f'<rect x="{x}" y="106" width="{len(text) * 8.2 + 14:.0f}" height="3" rx="1.5" class="{cls}f u{k}" opacity="0"/>'
        )
    # frame outline and crop outline
    parts.append(
        f'<rect x="{FX}" y="{FY}" width="{640 * FS}" height="{480 * FS}" rx="6" class="frame"/>'
    )
    parts.append(
        f'<text x="{FX}" y="{FY + 480 * FS + 18}" class="small">640×480 camera frame</text>'
    )
    parts.append(
        f'<rect x="{CX}" y="{CY}" width="{112 * CS}" height="{112 * CS}" rx="6" class="frame"/>'
    )
    for tx, ty in trace["template_112"]:
        parts.append(
            f'<circle cx="{CX + tx * CS:.1f}" cy="{CY + ty * CS:.1f}" r="6" class="slot"/>'
        )
    parts.append(f'<text x="{CX}" y="{CY + 112 * CS + 18}" class="small">112×112 template</text>')
    parts.append(
        f'<text x="{CX}" y="{CY + 112 * CS + 34}" class="small">dashed: template · dots: this face</text>'
    )
    # contribution axis
    parts.append(f'<line x1="{BX0 - 4}" x2="{BX0 + 218}" y1="{BY}" y2="{BY}" class="axis"/>')
    parts.append(
        f'<text x="{BX0}" y="{BY + 86}" class="small">512 dims in 32 groups: each bar adds</text>'
    )
    parts.append(
        f'<text x="{BX0}" y="{BY + 102}" class="small">its share of the cosine similarity</text>'
    )
    # decide plot: axes, Platt curve, threshold
    parts.append(
        f'<rect x="{DX0}" y="{DY1}" width="{DX1 - DX0}" height="{DY0 - DY1}" class="plot"/>'
    )
    pts = []
    for i in range(61):
        c = COS0 + (COS1 - COS0) * i / 60
        pts.append(f"{dx(c):.1f},{dy(1 / (1 + math.exp(-(a * c + b)))):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" class="curve"/>')
    parts.append(f'<line x1="{dx(thr):.1f}" x2="{dx(thr):.1f}" y1="{DY1}" y2="{DY0}" class="thr"/>')
    parts.append(
        f'<text x="{dx(thr) + 4:.1f}" y="{DY0 - 6}" class="small">unlock line {fmt(thr, 3)}</text>'
    )
    parts.append(f'<text x="{DX0}" y="{DY0 + 16}" class="small">cosine −0.2 → 1.0</text>')
    parts.append(f'<text x="{DX0 - 4}" y="{DY1 + 4}" class="small" text-anchor="end">P 1</text>')
    parts.append(f'<text x="{DX0 - 4}" y="{DY0}" class="small" text-anchor="end">0</text>')

    # ---------------------------------------------------------------- one group per attempt
    for kind, start in PHASES.items():
        t = trace[kind]
        g = kind[0]  # "g" or "i"
        ok = kind == "genuine"

        def at(p: float, start: float = start) -> float:
            return start + p

        # group visibility
        if ok:
            css.append(
                key(
                    "vg",
                    [
                        (0, "opacity:1"),
                        (44, "opacity:1"),
                        (47, "opacity:0"),
                        (97, "opacity:0"),
                        (100, "opacity:1"),
                    ],
                )
            )
        else:
            css.append(
                key(
                    "vi",
                    [
                        (0, "opacity:0"),
                        (47, "opacity:0"),
                        (50, "opacity:1"),
                        (94, "opacity:1"),
                        (97, "opacity:0"),
                        (100, "opacity:0"),
                    ],
                )
            )
        css.append(f".grp{g}{{animation:v{g} {T}s linear infinite}}")
        hidden = "" if ok else ' opacity="0"'  # the stranger starts hidden (and stays so at rest)
        parts.append(f'<g class="grp{g}"{hidden}>')

        # 1. detection box draws itself
        x, y, w, h = t["box"]
        per = 2 * (w + h) * FS
        parts.append(
            f'<rect x="{FX + x * FS:.1f}" y="{FY + y * FS:.1f}" width="{w * FS:.1f}" height="{h * FS:.1f}" '
            f'rx="4" class="box box{g}" style="stroke-dasharray:{per:.0f}"/>'
        )
        css.append(
            key(
                f"b{g}",
                [
                    (0, f"stroke-dashoffset:{per:.0f}"),
                    (at(2), f"stroke-dashoffset:{per:.0f}"),
                    (at(8), "stroke-dashoffset:0"),
                    (100, "stroke-dashoffset:0"),
                ],
            )
        )
        css.append(f".box{g}{{animation:b{g} {T}s ease-out infinite}}")
        # a neutral face outline in the detected box, so the landmarks read as a face
        cx_, cy_ = FX + (x + w / 2) * FS, FY + (y + h * 0.52) * FS
        parts.insert(
            len(parts) - 1,  # under the box just added
            f'<ellipse cx="{cx_:.1f}" cy="{cy_:.1f}" rx="{w * FS * 0.40:.1f}" ry="{h * FS * 0.44:.1f}" class="oval"/>',
        )
        parts.append(
            f'<text x="{FX + x * FS:.1f}" y="{FY + y * FS - 6:.1f}" class="small">face {fmt(t["detector_score"])}</text>'
        )

        # 2. landmarks appear on the face, then fly to the template
        for i, ((lx, ly), (ax, ay)) in enumerate(zip(t["landmarks"], t["aligned"], strict=True)):
            fx, fy = FX + lx * FS, FY + ly * FS
            tx, ty = CX + ax * CS, CY + ay * CS
            ox, oy = fx - tx, fy - ty
            name = f"l{g}{i}"
            delay = i * 0.4
            css.append(
                key(
                    name,
                    [
                        (0, f"transform:translate({ox:.1f}px,{oy:.1f}px);opacity:0"),
                        (at(8 + delay), f"transform:translate({ox:.1f}px,{oy:.1f}px);opacity:0"),
                        (at(10 + delay), f"transform:translate({ox:.1f}px,{oy:.1f}px);opacity:1"),
                        (at(13 + delay), f"transform:translate({ox:.1f}px,{oy:.1f}px);opacity:1"),
                        (at(19 + delay), "transform:translate(0px,0px);opacity:1"),
                        (100, "transform:translate(0px,0px);opacity:1"),
                    ],
                )
            )
            css.append(f".{name}{{animation:{name} {T}s cubic-bezier(.6,0,.3,1) infinite}}")
            parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="4.5" class="dot {name}"/>')

        # 3. the cosine, built from 32 real contributions
        for i, c in enumerate(t["contributions"]):
            h_px = abs(c) * BSCALE
            bx = BX0 + i * 6.8
            by = BY - h_px if c >= 0 else BY
            name = f"c{g}{i}"
            s = at(21 + i * 0.22)
            css.append(
                key(
                    name,
                    [
                        (0, "transform:scaleY(0)"),
                        (s, "transform:scaleY(0)"),
                        (s + 2.2, "transform:scaleY(1)"),
                        (100, "transform:scaleY(1)"),
                    ],
                )
            )
            origin = "bottom" if c >= 0 else "top"
            css.append(
                f".{name}{{transform-box:fill-box;transform-origin:{origin};animation:{name} {T}s ease-out infinite}}"
            )
            cls = "pos" if c >= 0 else "neg"
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="5" height="{max(h_px, 0.6):.1f}" rx="1.5" class="{cls} {name}"/>'
            )
        name = f"s{g}"
        css.append(
            key(
                name,
                [
                    (0, "opacity:0"),
                    (at(30), "opacity:0"),
                    (at(32), "opacity:1"),
                    (100, "opacity:1"),
                ],
            )
        )
        css.append(f".{name}{{animation:{name} {T}s linear infinite}}")
        parts.append(
            f'<text x="{BX0}" y="{BY - 96}" class="big {name}">cosine = Σ = {fmt(t["similarity"], 3)}</text>'
        )

        # 4. the dot rides the Platt curve from the left edge to the score
        sim = t["similarity"]
        fx, fy = dx(sim), dy(t["probability"])
        steps = []
        for k in range(11):
            c = COS0 + (sim - COS0) * k / 10
            px, py = dx(c), dy(1 / (1 + math.exp(-(a * c + b))))
            steps.append(
                (
                    at(32 + 7 * k / 10),
                    f"transform:translate({px - fx:.1f}px,{py - fy:.1f}px);opacity:1",
                )
            )
        name = f"d{g}"
        first = steps[0][1].replace("opacity:1", "opacity:0")
        css.append(
            key(
                name,
                [
                    (0, first),
                    (at(31.5), first),
                    *steps,
                    (100, "transform:translate(0px,0px);opacity:1"),
                ],
            )
        )
        css.append(f".{name}{{animation:{name} {T}s linear infinite}}")
        parts.append(
            f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="6.5" class="rider {"ok" if ok else "no"} {name}"/>'
        )

        # result badge
        name = f"r{g}"
        css.append(
            key(
                name,
                [
                    (0, "opacity:0;transform:scale(.92)"),
                    (at(39), "opacity:0;transform:scale(.92)"),
                    (at(41), "opacity:1;transform:scale(1)"),
                    (100, "opacity:1;transform:scale(1)"),
                ],
            )
        )
        css.append(
            f".{name}{{transform-box:fill-box;transform-origin:center;animation:{name} {T}s ease-out infinite}}"
        )
        word = "UNLOCKED" if ok else "LOCKED"
        rel = ">" if sim > thr else "<"
        p = t["probability"]
        ptxt = f"{p:.6f}" if p < 0.001 or p > 0.999 else f"{p:.3f}"
        parts.append(f'<g class="{name}">')
        parts.append(
            f'<rect x="{W / 2 - 250}" y="372" width="500" height="46" rx="23" class="badge {"ok" if ok else "no"}"/>'
        )
        parts.append(
            f'<text x="{W / 2}" y="401" text-anchor="middle" class="badgetext">'
            + html.escape(
                f"{'✓' if ok else '✕'} {word} · P(you) = {ptxt} · similarity {fmt(sim, 3)} {rel} {fmt(thr, 3)}"
            )
            + "</text>"
        )
        parts.append("</g>")
        parts.append("</g>")

    total = times["total_ms"]
    parts.append(
        f'<text x="{W / 2}" y="{H - 22}" text-anchor="middle" class="sub">Mac (Apple M4 Pro), Core ML fp16: '
        f"{fmt(total)} ms for the whole attempt. RGB only: a photo of the enrolled person would unlock it too.</text>"
    )

    style = f"""
svg{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.bg{{fill:#fcfcfb}} .title{{font-size:22px;font-weight:650;fill:#0b0b0b}}
.sub{{font-size:13px;fill:#52514e}} .head{{font-size:14px;font-weight:600;fill:#0b0b0b}}
.small{{font-size:11.5px;fill:#6b6a65}} .big{{font-size:17px;font-weight:650;fill:#0b0b0b}}
.frame{{fill:#f1f0ec;stroke:#d9d8d2}} .plot{{fill:#f6f5f2;stroke:#e1e0da}}
.slot{{fill:none;stroke:#c9c8c1;stroke-dasharray:2 2}}
.axis{{stroke:#bdbcb5}} .curve{{fill:none;stroke:#eda100;stroke-width:2.5}}
.thr{{stroke:#0b0b0b;stroke-width:1.5;stroke-dasharray:4 3}}
.box{{fill:none;stroke:#2a78d6;stroke-width:2.5}} .oval{{fill:#e6e4dd;stroke:#d2d0c8}} .dot{{fill:#eb6834;stroke:#fcfcfb;stroke-width:1.5}}
.pos{{fill:#1baf7a}} .neg{{fill:#eb6834}}
.c0f{{fill:#2a78d6}} .c1f{{fill:#eb6834}} .c2f{{fill:#1baf7a}} .c3f{{fill:#eda100}}
.rider{{stroke:#fcfcfb;stroke-width:2}} .rider.ok{{fill:{GREEN}}} .rider.no{{fill:{RED}}}
.badge.ok{{fill:{GREEN}}} .badge.no{{fill:{RED}}} .badgetext{{font-size:16px;font-weight:650;fill:#ffffff}}
{"".join(css)}
@media (prefers-color-scheme:dark){{
.bg{{fill:#1a1a19}} .title,.head,.big{{fill:#ffffff}} .sub{{fill:#c3c2b7}} .small{{fill:#a3a29a}}
.frame{{fill:#242423;stroke:#3a3a37}} .plot{{fill:#212120;stroke:#3a3a37}} .slot{{stroke:#55554f}}
.axis{{stroke:#55554f}} .curve{{stroke:#c98500}} .thr{{stroke:#ffffff}}
.box{{stroke:#3987e5}} .oval{{fill:#2e2e2c;stroke:#44443f}} .dot{{fill:#d95926;stroke:#1a1a19}} .pos{{fill:#199e70}} .neg{{fill:#d95926}}
.c0f{{fill:#3987e5}} .c1f{{fill:#d95926}} .c2f{{fill:#199e70}} .c3f{{fill:#c98500}}
.rider{{stroke:#1a1a19}} .rider.ok,.badge.ok{{fill:{GREEN_D}}} .rider.no,.badge.no{{fill:{RED_D}}}
.badgetext{{fill:#0b0b0b}}}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important}}}}
"""
    g, i = trace["genuine"], trace["impostor"]
    alt = (
        "Animation of one unlock attempt on real LFW data. The face is found, its five landmarks "
        "move onto the alignment template, the cosine similarity is built from 32 contributions, "
        f"and the Platt curve turns it into a probability. The enrolled person scores "
        f"{fmt(g['similarity'], 3)} and unlocks; a stranger scores {fmt(i['similarity'], 3)} and "
        f"stays locked. The unlock line is {fmt(thr, 3)}."
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="{html.escape(alt)}"><style>{style}</style>'
        + "".join(parts)
        + "</svg>\n"
    )
    return svg, alt
