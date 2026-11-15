#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
MODEL="distilbert-base-uncased"
DATA="$PROJ/data/en_tw_grand"
OUT="$PROJ/checkpoints/noun-marker"
VENV="$PROJ/.venv"
LOG_DIR="$PROJ/logs"

mkdir -p "$LOG_DIR" "$OUT"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ/repo"

echo "[$(date '+%F %T')] Training Nkrane-MT Inference Noun Phrase Marker on $MODEL..."

$VENV/bin/python training/train_marker.py \
    --model "$MODEL" \
    --dataset "$DATA" \
    --out "$OUT" \
    --epochs 10 \
    --batch-size 32 \
    --grad-accum 2 \
    --lr 3e-5 \
    --eval-steps 1000 \
    --early-stopping-patience 20 \
    --push ghanaopenai/nkrane-mt-marker-en \
    >> "$LOG_DIR/train_marker.log" 2>&1

echo "[$(date '+%F %T')] Marker training complete! Model saved to $OUT/final"
