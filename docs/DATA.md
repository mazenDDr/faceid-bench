# Data and licences

Nothing in this table is redistributed by this repository. `scripts/download_data.sh` fetches each
item from its source into `data/` (git-ignored); `data/SHA256SUMS` records what was used.

| Item | Used for | Source | Licence / terms |
|---|---|---|---|
| WIDER FACE val (3,226 images) + boxes | Detection accuracy (AP easy / medium / hard) | Hugging Face `CUHK-CSE/wider_face` | CC BY-NC-ND 4.0; non-commercial research |
| WIDER FACE `eval_tools` | Official easy / medium / hard face subsets | WIDER FACE project site | Distributed with the benchmark for evaluation |
| LFW (13,233 images, 5,749 people) + `pairs.txt` | Verification accuracy, TAR at fixed FAR, calibration | scikit-learn's figshare mirror of the UMass release | No formal licence; images of public figures from news sites, released for research |
| YuNet (`2023mar`, `2026may` ONNX) | Detector | OpenCV Zoo | MIT |
| MobileFaceNet / ResNet-50 (`w600k_mbf`, `w600k_r50`, WebFace600K) | Face embedding | InsightFace v0.7 release (`buffalo_s` / `buffalo_l`) | Models: non-commercial research only |
| SFace (`face_recognition_sface_2021dec.onnx`) | Face embedding | OpenCV Zoo | Apache 2.0 |
| SCRFD 500M / 10G with keypoints (`det_500m`, `det_10g` in `buffalo_s` / `buffalo_l`) | Detector | InsightFace v0.7 release | Models: non-commercial research only (InsightFace code is MIT) |

## Splits

- **Detection:** WIDER FACE val is the test set. Nothing is tuned on it; detector settings come from
  each model's published defaults.
- **Verification:** LFW identities are split into dev and test by a hash of the person's name
  (`faceid_bench.data.split_of`), so no person is on both sides. Thresholds and calibration are fit
  on dev; results are reported on test. At most 20 images per person are kept so that a few heavily
  photographed people do not dominate the genuine pairs.
- The official LFW 10-fold `pairs.txt` protocol is also reported, for comparison with published
  numbers.
