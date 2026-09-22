"""Face ID-style unlock on the Mac webcam: enroll your face, then unlock live.

  python scripts/demo_webcam.py            # camera 0; press E to enroll, Q to quit
  python scripts/demo_webcam.py --laya     # Laya makes the decision instead of Platt
  python scripts/demo_webcam.py --video clip.mp4 --enroll-frames 5 --headless   # no camera

The balanced pipeline (SCRFD-10G at 320 px + ResNet-50, Core ML fp16) and the unlock rule
measured on LFW: similarity above the dev-people threshold for 1 false accept in 100,000.
RGB only: there is no liveness check, so a photo of the enrolled person also unlocks it.
The face template is stored only on this machine, in data/demo/ (git-ignored).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from faceid_bench.latency import providers_for
from faceid_bench.pipeline import Config, Pipeline

TEMPLATE = Path("data/demo/template.npz")
GREEN, RED, WHITE, GREY = (80, 200, 60), (60, 60, 230), (255, 255, 255), (170, 170, 170)

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--video", help="read frames from a file instead of the camera")
parser.add_argument("--enroll-frames", type=int, default=0, help="enroll from the first N faces")
parser.add_argument("--headless", action="store_true", help="no window; print a summary")
parser.add_argument("--save-frames", help="write annotated frames here (checking the overlay)")
parser.add_argument("--laya", action="store_true", help="decide with Laya on the Mac GPU")
parser.add_argument("--laya-model", default="convaiinnovations/laya", help="or a local checkpoint")
parser.add_argument("--prior", type=float, default=0.5, help="share of attempts by the owner")
args = parser.parse_args()

balanced = json.loads(Path("results/pipeline.json").read_text())["configs"]["balanced"]
config = Config(**balanced["config"])
config = Config(config.name, config.detector, config.size, config.embedder, tuple(config.platt))
threshold = balanced["operating_points"]["1e-05"]["threshold_from_dev"]
pipe = Pipeline(config, providers_for("coreml", "all"))

decide_laya = None
if args.laya:
    from faceid_bench.laya_decision import QUESTION, LayaJudge, state

    judge = LayaJudge(args.laya_model, device="mps")
    no_det = np.zeros(15)

    def decide_laya(sim):
        return judge.agent.predict(state("note", sim, no_det, no_det), QUESTION)["answers"][
            "same_person"
        ]["noul"]


def enroll(frames: list[np.ndarray]) -> np.ndarray:
    """Average the embeddings of several frames into one template."""
    embeddings = [e for e in (pipe.embed(f) for f in frames) if e is not None]
    if not embeddings:
        raise RuntimeError("no face found in the enrollment frames")
    template = np.mean(embeddings, axis=0)
    template /= np.linalg.norm(template)
    TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TEMPLATE, template=template, frames=len(embeddings))
    return template


def draw(frame, result, unlocked, fps):
    h, w = frame.shape[:2]
    if result.get("face_row") is not None:
        x, y, bw, bh = result["face_row"][:4].astype(int)
        color = GREEN if unlocked else RED
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        for px, py in result["face_row"][4:14].reshape(5, 2).astype(int):
            cv2.circle(frame, (px, py), 2, color, -1)
    status = "UNLOCKED" if unlocked else "LOCKED"
    cv2.rectangle(frame, (0, 0), (w, 118), (20, 20, 20), -1)
    cv2.putText(
        frame, status, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN if unlocked else RED, 2
    )
    if result.get("face"):
        who = "Laya" if args.laya else "Platt"
        text = (
            f"P(you) {result['probability']:.3f} ({who})   "
            f"similarity {result['similarity']:.3f} (unlock above {threshold:.3f})"
        )
        cv2.putText(frame, text, (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    t = result["times"]
    stages = "  ".join(
        f"{k[:-3]} {t[k]:.3f}" if t[k] < 0.1 else f"{k[:-3]} {t[k]:.1f}"
        for k in ("detect_ms", "align_ms", "embed_ms", "decide_ms")
        if k in t
    )
    cv2.putText(
        frame,
        f"{stages}   total {t['total_ms']:.1f} ms   {fps:.0f} fps",
        (14, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        1,
    )
    cv2.putText(
        frame,
        "RGB only: a photo of you would also unlock this.  E enroll  Q quit",
        (14, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        GREY,
        1,
    )


source = cv2.VideoCapture(args.video if args.video else args.camera)
if not source.isOpened():
    raise SystemExit(
        "camera not available: allow camera access for your terminal in System Settings > "
        "Privacy & Security > Camera, or pass --video"
    )
template = np.load(TEMPLATE)["template"] if TEMPLATE.exists() and not args.enroll_frames else None
enroll_buffer, log, last = [], [], time.perf_counter()
while True:
    ok, frame = source.read()
    if not ok:
        break
    if args.enroll_frames and template is None:
        if pipe.embed(frame) is not None:
            enroll_buffer.append(frame)
        if len(enroll_buffer) >= args.enroll_frames:
            template = enroll(enroll_buffer)
        continue
    result = (
        pipe.verify(frame, template, args.prior)
        if template is not None
        else {"face": False, "times": {"total_ms": 0.0}}
    )
    if decide_laya is not None and result.get("face"):
        t = time.perf_counter()
        result["probability"] = decide_laya(result["similarity"])
        result["times"]["decide_ms"] = (time.perf_counter() - t) * 1e3
        result["times"]["total_ms"] += result["times"]["decide_ms"]
    unlocked = bool(result.get("face")) and (
        result["probability"] > 0.5 if args.laya else result["similarity"] > threshold
    )
    now = time.perf_counter()
    fps, last = 1 / max(now - last, 1e-6), now
    log.append({"unlocked": unlocked, "face": bool(result.get("face")), **result["times"]})
    if args.headless and not args.save_frames:
        continue
    draw(frame, result, unlocked, fps)
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{args.save_frames}/{len(log):04d}.jpg", frame)
    if args.headless:
        continue
    cv2.imshow("faceid-bench", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("e"):
        frames = []
        while len(frames) < 5:
            ok, f = source.read()
            if ok and pipe.embed(f) is not None:
                frames.append(f)
        template = enroll(frames)

source.release()
cv2.destroyAllWindows()
if log:
    summary = {
        "attempts": len(log),
        "unlocked": sum(r["unlocked"] for r in log),
        "no_face": sum(not r["face"] for r in log),
        # run-length pattern of the decisions, e.g. "U25 L20" = 25 unlocked then 20 locked
        "pattern": " ".join(
            f"{'U' if run[0] else 'L'}{len(run)}"
            for run in np.split(
                np.array([r["unlocked"] for r in log]),
                np.flatnonzero(np.diff([r["unlocked"] for r in log])) + 1,
            )
        ),
        "median_ms": {
            k: round(statistics.median(r[k] for r in log if k in r), 3)
            for k in ("detect_ms", "align_ms", "embed_ms", "decide_ms", "total_ms")
            if any(k in r for r in log)
        },
    }
    print(json.dumps(summary, indent=2))
