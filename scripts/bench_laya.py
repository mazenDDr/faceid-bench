"""Time one Laya decision (the 'rich' state, one noul question) on this machine's devices.

Same protocol as the model timings: warm-up, batch 1, median of fresh sessions.
Writes outputs/laya/latency_<machine>.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from faceid_bench.latency import machine_label, measure, median_session, summarize
from faceid_bench.laya_decision import QUESTION, LayaJudge, state

devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["mps", "cpu"]
det = np.zeros(15)
det[2], det[14] = 90, 0.8
example = state("rich", 0.41, det, det)
result = {"machine": machine_label(), "model": "convaiinnovations/laya", "devices": {}}
for device in devices:
    start = time.perf_counter()
    agent = LayaJudge(device=device).agent
    load_s = time.perf_counter() - start

    def run(agent=agent, device=device):
        agent.predict(example, QUESTION)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    sessions = [summarize(measure(run, 10, 50)) for _ in range(3)]
    result["devices"][device] = {
        **median_session(sessions),
        "session_p50_ms": [s["p50_ms"] for s in sessions],
        "load_s": round(load_s, 1),
        "runs": 50,
        "warmup": 10,
    }
    print(device, result["devices"][device])
    del agent
# The alternative decision: Platt on the cosine (two numbers from the calibration step)
a, b, sim = 37.5409, -8.4249, 0.41


def platt_decision():
    return 1 / (1 + np.exp(-(a * sim + b)))


# too fast for per-call timing: time 100,000 calls per session, report the per-call median
per_call_us = []
for _ in range(3):
    for _ in range(1000):
        platt_decision()
    start = time.perf_counter()
    for _ in range(100_000):
        platt_decision()
    per_call_us.append((time.perf_counter() - start) / 100_000 * 1e6)
result["platt_cosine"] = {
    "per_call_us": round(sorted(per_call_us)[1], 3),
    "session_us": [round(v, 3) for v in per_call_us],
    "calls_per_session": 100_000,
}
print("platt", result["platt_cosine"])
out = Path(f"outputs/laya/latency_{result['machine']}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(f"saved {out}")
