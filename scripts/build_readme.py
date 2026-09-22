"""Fill the README's GENERATED blocks and write docs/assets/hero.svg from results/*.json."""

from pathlib import Path

from faceid_bench import report

pipeline = report.load("pipeline")
svg, alt = report.hero_svg(pipeline)
Path("docs/assets").mkdir(parents=True, exist_ok=True)
Path("docs/assets/hero.svg").write_text(svg)

balanced = pipeline["configs"]["balanced"]
tagline = (
    f"<b>A Face ID-style unlock in {report.fmt_ms(balanced['mac_coreml_all_fp16']['total_ms'])} ms "
    f"on a Mac, accepting the right person {balanced['operating_points']['1e-05']['test_tar']:.2%} "
    f"of the time at 1 false accept in 100,000.</b><br>\n"
    "Every model choice measured, with intervals. And a fair test of Laya."
)
blocks = {
    "HERO": (
        f'<p align="center">\n  <img src="docs/assets/hero.svg" width="100%" alt="{alt}">\n</p>'
    ),
    "TAGLINE": tagline,
    "PIPELINE": report.pipeline_table(pipeline),
    "LAYA": report.laya_table(report.load("laya")),
    "DETECTORS": report.detector_table(report.load("detectors")),
    "VERIFICATION": report.verification_table(report.load("verification")),
    "CALIBRATION": report.calibration_table(report.load("calibration")),
    "FAILURES": report.failure_table(report.load("failures")),
}
text = Path("README.md").read_text()
for name, body in blocks.items():
    text = report.replace_block(text, name, body)
Path("README.md").write_text(text)
print(f"README.md: {len(blocks)} blocks; docs/assets/hero.svg: {len(svg):,} bytes")
