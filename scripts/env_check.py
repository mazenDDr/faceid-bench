"""Print and save what this machine can run (`make env-check`, or `./gpu exec` on gpu-box)."""

import json
from pathlib import Path

from faceid_bench.env import collect

report = collect()
out = Path("outputs") / f"env_{report['host']}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"saved {out}")
