#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
MODEL="google/umt5-base"
DIRECTION="en-tw"
DATA="$PROJ/data/en_tw_grand"
OUT="$PROJ/checkpoints/umt5-base-en2tw"
VENV="$PROJ/.venv"
LOG_DIR="$PROJ/logs"

mkdir -p "$LOG_DIR" "$OUT"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"

echo "=== [1/1] Fine-tuning $MODEL ($DIRECTION) on Nkrane-MT entity-grounded dataset ==="
$VENV/bin/python training/train_seq2seq.py \
    --model "$MODEL" \
    --data "$DATA" \
    --direction "$DIRECTION" \
    --out "$OUT" \
    --epochs 30 \
    --batch-size 16 \
    --grad-accum 2 \
    --lr 2e-4 \
    --max-src 512 \
    --max-tgt 512 \
    --eval-steps 1000 \
    --early-stopping-patience 40 \
    --gen-eval-samples 300 \
    --push ghanaopenai/nkrane-mt-umt5-base-en2tw \
    >> "$LOG_DIR/train_umt5_base_en2tw.log" 2>&1

echo "=== uMT5 Base ($DIRECTION) training complete! ==="
