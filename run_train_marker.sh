#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

MODEL="distilbert-base-uncased"

# Use local cache if available on H200, otherwise load directly from Hugging Face Hub
HF_DATASET="ghanaopenai/nkrane-mt-parallel-corpus-eng-twi"
LOCAL_DATA="/mnt/volume_d2wey28/projects/gloss-mt-2/data/en_tw_grand"
if [ -d "$LOCAL_DATA" ]; then
    DATA="$LOCAL_DATA"
else
    DATA="${DATA:-$HF_DATASET}"
fi

OUT="${OUT:-$DIR/checkpoints/noun-marker}"
LOG_DIR="${LOG_DIR:-$DIR/logs}"
mkdir -p "$LOG_DIR" "$OUT"

# Select Python binary
if [ -x "/mnt/volume_d2wey28/projects/gloss-mt-2/.venv/bin/python" ]; then
    PYTHON="/mnt/volume_d2wey28/projects/gloss-mt-2/.venv/bin/python"
elif [ -x "$DIR/.venv/bin/python" ]; then
    PYTHON="$DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

if [ -d "/usr/local/cuda-13.0/compat" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Training Nkrane-MT Inference Noun Phrase Marker on $MODEL ($DATA) ==="

$PYTHON training/train_marker.py \
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

echo "=== Marker training complete! Model saved to $OUT/final ==="
