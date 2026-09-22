<!-- BEGIN GENERATED HERO -->
<p align="center">
  <img src="docs/assets/hero.svg" width="100%" alt="One unlock attempt on a Mac (Apple M4 Pro): finding the face, aligning it, embedding it and deciding takes 4.75 ms. With Laya making the decision it takes 35.5 ms, of which Laya is 30.0 ms.">
</p>
<!-- END GENERATED HERO -->

<h1 align="center">faceid-bench</h1>

<p align="center">
<!-- BEGIN GENERATED TAGLINE -->
<b>A Face ID-style unlock in 4.75 ms on a Mac, accepting the right person 99.48% of the time at 1 false accept in 100,000.</b><br>
Every model choice measured, with intervals. And a fair test of Laya.
<!-- END GENERATED TAGLINE -->
</p>

<p align="center">
  <a href="#results-at-a-glance"><b>Results</b></a>
  &nbsp;·&nbsp;
  <a href="#where-laya-fits"><b>Laya</b></a>
  &nbsp;·&nbsp;
  <a href="#how-each-stage-was-chosen"><b>How each stage was chosen</b></a>
  &nbsp;·&nbsp;
  <a href="#what-goes-wrong"><b>What goes wrong</b></a>
  &nbsp;·&nbsp;
  <a href="#reproduce"><b>Reproduce</b></a>
  &nbsp;·&nbsp;
  <a href="https://mazenddr.github.io/faceid-bench/"><b>Interactive page</b></a>
</p>

---

The iPhone's Face ID does two jobs: it finds your face, then decides whether it is *you*. This
project rebuilds that with open models: **detect → align → embed → compare → calibrated
probability**. Every part was picked by measurement: accuracy with 95% intervals, compared on the
same data, and speed timed on an Apple M4 Pro (Core ML, Neural Engine) and an RTX 5060 Ti.

It also puts [Laya](https://github.com/NandhaKishorM/laya), a fast "decision engine" language
model, to the test in the one step it can do: the final yes/no.

**What this is not.** Face ID uses an infrared depth camera to reject photos and masks. This
project uses ordinary RGB images, so it has **no liveness or spoof protection**. Accuracy is
measured on LFW photos of public figures, not live camera frames, and nothing here ran on an
iPhone.

## Results at a glance

<!-- BEGIN GENERATED PIPELINE -->
| Pipeline | Models | Accepts the right person, TAR @ FAR 1e-5 | Mac p50 / p95 per attempt |
|---|---|---|---:|
| Fast | SCRFD-500M @ 320 px + MobileFaceNet | 0.9857 [0.9780, 0.9920] | 2.27 / 3.95 ms |
| **Balanced** | SCRFD-10G @ 320 px + ResNet-50 | 0.9948 [0.9900, 0.9985] | 4.75 / 5.81 ms |
| Accurate | SCRFD-10G @ 640 px + ResNet-50 | 0.9942 [0.9891, 0.9981] | 10.4 / 12.0 ms |
| Balanced, **Laya decides** | same, Laya zero-shot on the Mac GPU | 0.9455 @ FAR 1e-3 | 35.5 / 37.1 ms |

*Accuracy: LFW, 2,865 people never seen while tuning. The threshold is fixed on separate dev people for 1 false accept in 100,000, then applied to 13,183 same-person and 16,183,403 different-person pairs. Time: one attempt on a 640x480 frame on a Mac (Apple M4 Pro), Core ML fp16, median of 3 sessions. The Laya row is measured at a looser setting (1 in 1,000).*
<!-- END GENERATED PIPELINE -->

**Balanced** is the pick: as accurate as the 640 px version (the difference is not significant) in
under half the time.

**[Pick your own unlock threshold](https://mazenddr.github.io/faceid-bench/)** — an interactive page
(`site/index.html`, built by `make readme`) that shows every test pair's
score and lets you move the unlock line and the prior, with the true-accept rate, strangers per
million and the calibrated probability updating live.

## Where Laya fits

Laya reads **text or JSON, not pixels**, so it cannot find a face in an image. It can do the last
step: read the pipeline's numbers and answer *"same person?"* with a probability. That step was
measured head-to-head against Platt scaling (two numbers fit on dev people), on the same pairs.

<!-- BEGIN GENERATED LAYA -->
| Decision step | AUC | TAR @ FAR 1e-3 | Cllr (lower is better) | vs Platt, paired | Mac time per decision |
|---|---:|---:|---:|---|---:|
| Platt on the cosine (2 numbers) | 0.9995 | 0.9970 | 0.0152 | — | 0.12 µs |
| Laya zero-shot, number only | 0.8251 | 0.0010 | 1.9693 | +1.954 [1.937, 1.966] | 37.6 ms |
| Laya zero-shot, number + one sentence | 0.9908 | 0.9455 | 0.2622 | +0.246 [0.228, 0.264] | 37.6 ms |
| Laya zero-shot, rich JSON | 0.7779 | 0.0190 | 0.8617 | +0.846 [0.815, 0.879] | 37.6 ms |
| **Laya fine-tuned** on dev people | 0.9987 | 0.9965 | 0.0183 | +0.003 [-0.002, 0.008] (tie) | 37.6 ms¹ |

*2,000 same-person and 20,000 different-person pairs of test people; 95% intervals resample people. Time: one decision on the Mac GPU (Laya) or CPU (Platt), median of 3 sessions. ¹ Timed with the base model; the fine-tuned model has the same architecture.*
<!-- END GENERATED LAYA -->

- **Zero-shot**, Laya barely reads the similarity number; one sentence explaining the number helps a
  lot, extra JSON fields make it worse.
- **Fine-tuned** with Laya's own recipe on dev people, it **ties** Platt and has the best-calibrated
  probabilities in the project. Both read the same number, so a tie is the ceiling.
- The cost is the catch: one Laya decision takes longer than the whole face pipeline, many times
  over. It is fast for a 421M-parameter language model, and its published speed claim holds, but
  here the decision is two multiplications.

## How each stage was chosen

**1. Find the face.** WIDER FACE validation set; our scorer reproduces OpenCV's published YuNet
result before comparing anything.

<!-- BEGIN GENERATED DETECTORS -->
| Detector | Easy AP | Medium AP | Hard AP | Mac, model only |
|---|---|---|---|---:|
| YuNet 640 | 0.884 [0.874, 0.894] | 0.866 [0.854, 0.876] | 0.750 [0.732, 0.768] | 1.31 ms |
| SCRFD-500M 640 | 0.909 [0.900, 0.918] | 0.884 [0.873, 0.894] | 0.693 [0.668, 0.717] | 1.69 ms |
| SCRFD-10G 640 | 0.951 [0.944, 0.956] | 0.936 [0.929, 0.943] | 0.824 [0.805, 0.840] | 5.26 ms |
| SCRFD-500M 320 | 0.854 [0.842, 0.866] | 0.762 [0.747, 0.778] | 0.378 [0.350, 0.408] | 0.69 ms |
| SCRFD-10G 320 | 0.924 [0.915, 0.932] | 0.866 [0.854, 0.878] | 0.513 [0.483, 0.545] | 1.61 ms |
| YuNet 320 | 0.786 [0.771, 0.800] | 0.666 [0.648, 0.684] | 0.305 [0.281, 0.333] | 0.90 ms |
<!-- END GENERATED DETECTORS -->

**2. Is it the same person?** LFW, split by person: thresholds on dev people, results on test
people. The official 10-fold accuracy is shown next to the published figure as a reproduction check.

<!-- BEGIN GENERATED VERIFICATION -->
| Face model | LFW 10-fold (published) | TAR @ 1e-3 | TAR @ 1e-4 | TAR @ 1e-5 | Mac, model only |
|---|---|---:|---:|---:|---:|
| ResNet-50 (WebFace600K) | 0.9982 (0.9983) | 0.9964 | 0.9959 | 0.9942 | 1.75 ms |
| MobileFaceNet (WebFace600K) | 0.9950 (0.9970) | 0.9944 | 0.9917 | 0.9844 | 0.40 ms |
| SFace | 0.9942 (0.9940) | 0.9919 | 0.9871 | 0.9694 | 3.77 ms |
<!-- END GENERATED VERIFICATION -->

**3. Turn the score into a probability.** Cllr is a proper scoring rule (0 is perfect, 1 is "no
idea"). Reading a cosine as a probability is badly wrong; two numbers fix it.

<!-- BEGIN GENERATED CALIBRATION -->
| Score → probability | Cllr | ECE |
|---|---|---:|
| Read the cosine as a probability | 0.6346 [0.6305, 0.6383] | 0.3305 |
| **Platt** (2 numbers) | 0.0199 [0.0042, 0.0479] | 0.0012 |
| Isotonic | 0.0183 [0.0041, 0.0454] | 0.0012 |
| Best possible for this model | 0.0167 | — |
<!-- END GENERATED CALIBRATION -->

## What goes wrong

Every mistake of the balanced pipeline on test people, traced to the first stage that broke.
Thresholds for "small", "blurred", "dark" and "extreme pose" are dev-side percentiles, never picked
from the failures.

<!-- BEGIN GENERATED FAILURES -->
| Stage that broke | Rejected the right person (69) | Accepted the wrong person (86) |
|---|---:|---:|
| data: wrong label | 39 | 20 |
| embed: model | 22 | 41 |
| detect: small face | 3 | 4 |
| align: extreme pose | 2 | 7 |
| image: dark or washed out | 2 | 5 |
| image: blurred | 1 | 9 |
| fixed by the 640 px detector | 1 | 10 |
<!-- END GENERATED FAILURES -->

Most rejections of the right person come from **mistakes in LFW itself**, confirmed by looking at
the photos: one "Mahmoud Abbas" image shows another man, one "Erdoğan" image shows Abdullah Gül, one
photo appears under two names, and two different men are both "Jim O'Brien". Main results keep the
official labels; a clearly marked sensitivity run removes these.

**Other things that were checked and turned out wrong** (and are fixed):
- Core ML's own "compute plan" reported CPU for every layer on macOS 27 while the Neural Engine was
  doing the work. Placement is judged by timing each compute unit instead.
- onnxruntime's CUDA provider silently ran on the CPU unless PyTorch happened to be imported first.
  Every timing row records the provider that actually ran.
- CUDA sessions on the RTX box come up in two speed modes, so every timing is the median of several
  fresh sessions.
- Fixed-size model exports kept stale output shapes, which changed how Core ML split the model.

## Try it on your webcam

On a Mac with the models in `models/` (see [Reproduce](#reproduce)):

```bash
make demo                                          # press E to enroll your face, Q to quit
PYTHONPATH=src python scripts/demo_webcam.py --laya   # same, with Laya making the decision
```

The window shows LOCKED / UNLOCKED, the calibrated probability, and each stage's time live. It uses
the balanced pipeline and the unlock rule measured above. Your face template is saved only on your
machine, in `data/demo/`. **A photo of you will also unlock it**: there is no depth camera.

The first time, macOS asks to let your terminal use the camera (System Settings → Privacy & Security
→ Camera). Without a camera, `--video clip.mp4 --enroll-frames 5 --headless` runs the same loop on a
file and prints a summary.

## Reproduce

Code is edited on the Mac; data, training and CUDA timing run on a GPU machine (`./gpu` wraps the
rsync/ssh/tmux calls). Core ML timing runs on the Mac.

```bash
make check                                     # lint + tests
bash scripts/download_data.sh                  # on the GPU machine: WIDER FACE val, LFW
python scripts/eval_wider.py scrfd_10g_kps     # detector AP (GPU machine)
python scripts/embed_lfw.py r50_w600k          # embeddings (GPU machine)
python scripts/eval_verification.py r50_w600k  # TAR at fixed FAR, person bootstrap
python scripts/eval_calibration.py r50_w600k   # Cllr / ECE
python scripts/bench_pipeline.py --laya        # whole pipeline on the Mac
make pull && make readme                       # results/*.json -> this README
```

Every number in this README is generated from `results/*.json` by `make readme`.

## Data and licences

Datasets are downloaded from their sources and never redistributed; see
[docs/DATA.md](docs/DATA.md). Code: MIT. Models keep their own licences: YuNet (MIT), SFace
(Apache 2.0), Laya (Apache 2.0), and the InsightFace SCRFD / ArcFace models (non-commercial research
only).
