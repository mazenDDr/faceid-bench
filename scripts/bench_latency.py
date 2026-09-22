"""Time one model on several backends and append one JSON line per backend.

Examples:
  python scripts/bench_latency.py models/yunet.onnx --onnx cpu coreml:ane coreml:cpu
  python scripts/bench_latency.py models/yunet.onnx --onnx cpu cuda tensorrt  (on gpu-box)
  python scripts/bench_latency.py models/probe.mlpackage --coreml cpu gpu ane
"""

import argparse
from dataclasses import asdict

from faceid_bench.latency import append, machine_label, time_coreml, time_onnx

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--onnx", nargs="*", default=[], help="backend or coreml:<unit>")
parser.add_argument("--coreml", nargs="*", default=[], help="cpu | gpu | ane | all")
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--runs", type=int, default=200)
parser.add_argument("--sessions", type=int, default=3, help="fresh sessions per backend")
parser.add_argument("--out", help="default: outputs/latency_<machine>.jsonl")
args = parser.parse_args()
# One file per machine: `./gpu pull` copies outputs/ back and would overwrite a shared name.
args.out = args.out or f"outputs/latency_{machine_label()}.jsonl"

results = []
for spec in args.onnx:
    backend, _, unit = spec.partition(":")
    results.append(
        time_onnx(args.model, backend, unit or "all", args.warmup, args.runs, args.sessions)
    )
for unit in args.coreml:
    results.append(time_coreml(args.model, unit, args.warmup, args.runs))

for r in results:
    append(r, args.out)
    row = asdict(r)
    print(
        f"{row['machine']:12s} {row['backend']:10s} {row['unit']:9s} "
        f"p50 {row['p50_ms']:8.3f}  p95 {row['p95_ms']:8.3f}  load {row['load_ms']:7.1f} ms"
        f"  sessions {row['session_p50_ms']}" + (f"  !! {row['note']}" if row["note"] else "")
    )
print(f"appended {len(results)} rows to {args.out}")
