# Data and licences

Nothing in this table is redistributed by this repository. `scripts/download_data.sh` fetches each
item from its source into `data/` (git-ignored); `data/SHA256SUMS` records what was used.

| Item | Used for | Source | Licence / terms |
|---|---|---|---|
| WIDER FACE val (3,226 images) + boxes | Detection accuracy (AP easy / medium / hard) | Hugging Face `CUHK-CSE/wider_face` | CC BY-NC-ND 4.0; non-commercial research |
| WIDER FACE `eval_tools` | Official easy / medium / hard face subsets | WIDER FACE project site | Distributed with the benchmark for evaluation |
| LFW (13,233 images, 5,749 people) + `pairs.txt` | Verification accuracy, TAR at fixed FAR, calibration | scikit-learn's figshare mirror of the UMass release | No formal licence; images of public figures from news sites, released for research |
| YuNet (`face_detection_yunet_2023mar.onnx`) | Detector baseline, latency smoke test | OpenCV Zoo | MIT |

## Splits

- **Detection:** WIDER FACE val is the test set. Nothing is tuned on it; detector settings come from
  each model's published defaults.
- **Verification:** LFW identities are split into dev and test by a hash of the person's name
  (`faceid_bench.data.split_of`), so no person is on both sides. Thresholds and calibration are fit
  on dev; results are reported on test. At most 20 images per person are kept so that a few heavily
  photographed people do not dominate the genuine pairs.
- The official LFW 10-fold `pairs.txt` protocol is also reported, for comparison with published
  numbers.
