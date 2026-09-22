#!/usr/bin/env bash
# Download the benchmark data into data/ (run on gpu-box). Safe to re-run: finished files are kept.
# Licences are summarised in docs/DATA.md; none of these files may be redistributed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/wider_face data/lfw

# WIDER FACE validation set + boxes (CC BY-NC-ND 4.0) and the official eval subsets (easy/medium/hard).
hf download CUHK-CSE/wider_face data/WIDER_val.zip data/wider_face_split.zip \
  --repo-type dataset --local-dir data/wider_face
[ -f data/wider_face/eval_tools.zip ] || curl -sfL -o data/wider_face/eval_tools.zip \
  http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip

# LFW original images + official pair lists, from scikit-learn's figshare mirror.
fetch() { [ -f "data/lfw/$2" ] || curl -sfL -A "Mozilla/5.0" -o "data/lfw/$2" "https://ndownloader.figshare.com/files/$1"; }
fetch 5976018 lfw.tgz
fetch 5976006 pairs.txt
fetch 5976012 pairsDevTrain.txt
fetch 5976009 pairsDevTest.txt

for z in data/wider_face/data/WIDER_val.zip data/wider_face/data/wider_face_split.zip data/wider_face/eval_tools.zip; do
  unzip -qn "$z" -d data/wider_face/
done
[ -d data/lfw/lfw ] || tar -xzf data/lfw/lfw.tgz -C data/lfw/

sha256sum data/wider_face/data/*.zip data/wider_face/eval_tools.zip data/lfw/*.tgz data/lfw/*.txt > data/SHA256SUMS
cat data/SHA256SUMS
echo "download finished"
