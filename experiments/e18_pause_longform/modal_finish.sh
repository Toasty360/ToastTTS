#!/bin/bash
# e18: download word JSONs from Modal volume, sort into train/val, then
# build labels, analyze gaps, and train locally (CPU, minutes).
set -u
E18=~/workspace/toasttts/experiments/e18_pause_longform
M=~/workspace/.modal-venv/bin/modal
PY=~/workspace/.pause-venv/bin/python
cd "$E18"

echo "=== downloading word JSONs from e18-words volume ==="
rm -rf data/words_modal && mkdir -p data/words_modal
for f in $($M volume ls e18-words / 2>/dev/null); do
  $M volume get --force e18-words "/$f" "data/words_modal/$f" >/dev/null 2>&1 || echo "FAIL $f"
done
echo "downloaded: $(ls data/words_modal/*.json 2>/dev/null | wc -l) json files"

echo "=== sorting into train/val ==="
mkdir -p data/words_train data/words_val
for f in data/words_modal/*.json; do
  book=$(basename "$f" | cut -d_ -f1)
  if [ "$book" = "heartdarkness" ]; then
    cp "$f" data/words_val/
  else
    cp "$f" data/words_train/
  fi
done
echo "train: $(ls data/words_train/*.json | wc -l) / val: $(ls data/words_val/*.json | wc -l)"

echo "=== labels ==="
$PY build_labels.py --train-json data/words_train --val-json data/words_val --outdir data/labels

echo "=== gap analysis (the key diagnostic) ==="
$PY analyze_gaps.py --labels data/labels

echo "=== training ==="
$PY train.py --train data/labels/examples_train.jsonl --val data/labels/examples_val.jsonl \
  --out data/model --epochs 25
echo "=== pipeline done: $(date) ==="
