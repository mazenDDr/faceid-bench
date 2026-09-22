import json
import xml.dom.minidom
from pathlib import Path

import pytest

from faceid_bench import report


def test_replace_block_swaps_only_the_marked_text():
    text = "a\n<!-- BEGIN GENERATED X -->\nold\n<!-- END GENERATED X -->\nb"
    out = report.replace_block(text, "X", "new")
    assert out == "a\n<!-- BEGIN GENERATED X -->\nnew\n<!-- END GENERATED X -->\nb"
    assert report.replace_block(out, "X", "new") == out  # idempotent
    with pytest.raises(KeyError):
        report.replace_block(text, "Y", "new")


@pytest.mark.skipif(not Path("results/pipeline.json").exists(), reason="no results")
def test_hero_reads_its_numbers_from_the_results():
    pipeline = json.loads(Path("results/pipeline.json").read_text())
    svg, alt = report.hero_svg(pipeline)
    xml.dom.minidom.parseString(svg)  # well-formed
    balanced = pipeline["configs"]["balanced"]["mac_coreml_all_fp16"]["total_ms"]
    laya = pipeline["configs"]["balanced+laya"]["mac_coreml_all_fp16"]["total_ms"]
    assert f"{report.fmt_ms(balanced)} ms" in svg and f"{report.fmt_ms(laya)} ms" in alt
    assert "prefers-reduced-motion" in svg and "prefers-color-scheme:dark" in svg


def test_readme_is_up_to_date_with_results():
    """`make readme` must have been run after the last results change."""
    if not Path("results/pipeline.json").exists():
        pytest.skip("no results")
    text = Path("README.md").read_text()
    pipeline = report.load("pipeline")
    assert report.pipeline_table(pipeline) in text
    assert report.laya_table(report.load("laya")) in text


def test_built_site_embeds_valid_data():
    site = Path("site/index.html")
    if not site.exists():
        pytest.skip("site not built")
    html = site.read_text()
    assert "/*__DATA__*/" not in html
    raw = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    data = json.loads(raw.replace("<\\/", "</"))
    assert sum(data["scores"]["genuine"]) > 0 and len(data["presets"]) == 3
    assert data["laya"][0]["label"].startswith("Two numbers")
