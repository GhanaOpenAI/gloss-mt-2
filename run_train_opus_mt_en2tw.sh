#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
MODEL="Helsinki-NLP/opus-mt-en-tw"
DIRECTION="en-tw"
DATA="$PROJ/data/en_tw_grand"
OUT="$PROJ/checkpoints/opus-mt-en2tw"
VENV="$PROJ/.venv"
LOG_DIR="$PROJ/logs"

mkdir -p "$LOG_DIR" "$OUT"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"

echo "=== [1/1] Fine-tuning $MODEL ($DIRECTION) on GLOSS-MT-2 preserved dataset ==="
$VENV/bin/python training/train_seq2seq.py \
    --model "$MODEL" \
    --data "$DATA" \
    --direction "$DIRECTION" \
    --out "$OUT" \
    --epochs 20 \
    --batch-size 32 \
    --grad-accum 1 \
    --lr 1e-4 \
    --max-src 512 \
    --max-tgt 512 \
    --eval-steps 500 \
    --early-stopping-patience 4 \
    >> "$LOG_DIR/train_opus_mt_en2tw.log" 2>&1

echo "=== OPUS-MT ($DIRECTION) training complete! ==="
