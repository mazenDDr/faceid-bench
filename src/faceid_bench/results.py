"""Shared helpers for building the committed results files from raw run outputs."""

from __future__ import annotations

import json
from pathlib import Path


def latency_rows(outputs: Path = Path("outputs")) -> list[dict]:
    """All timing rows from every machine, minus any that silently fell back to another provider."""
    rows = []
    for f in sorted(outputs.glob("latency_*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return [r for r in rows if not r["note"]]


def best(
    rows: list[dict], model: str, machine: str, fp16_ok: bool, backends: set[str]
) -> dict | None:
    """Fastest median-session p50 for `model` on `machine`; fp16 only where its accuracy was
    checked against fp32."""
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


def latency_summary(rows: list[dict], model: str, fp16_ok: bool) -> dict:
    return {
        "mac-m4pro_best": best(rows, model, "mac-m4pro", fp16_ok, {"ort-coreml", "ort-cpu"}),
        "mac-m4pro_cpu": best(rows, model, "mac-m4pro", False, {"ort-cpu"}),
        "rtx5060ti_cuda": best(rows, model, "rtx5060ti", fp16_ok, {"ort-cuda"}),
        "rtx5060ti_cpu": best(rows, model, "rtx5060ti", False, {"ort-cpu"}),
    }


def ms(v: dict | None, show_setting: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v['p50_ms']:.2f}" + (f" ({v['setting'].split(':', 1)[1]})" if show_setting else "")
