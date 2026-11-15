#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
MODEL="facebook/nllb-200-distilled-600M"
DIRECTION="en-tw"
DATA="$PROJ/data/en_tw_grand"
OUT="$PROJ/checkpoints/nllb-600m-en2tw"
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
    --batch-size 16 \
    --grad-accum 2 \
    --lr 2e-4 \
    --max-src 512 \
    --max-tgt 512 \
    --eval-steps 500 \
    --early-stopping-patience 4 \
    >> "$LOG_DIR/train_nllb_600m_en2tw.log" 2>&1

echo "=== NLLB 600M ($DIRECTION) training complete! ==="
