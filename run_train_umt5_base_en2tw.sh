#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

MODEL="google/umt5-base"
DIRECTION="en-tw"

# Use local cache if available on H200, otherwise load directly from Hugging Face Hub
HF_DATASET="ghanaopenai/nkrane-mt-parallel-corpus-eng-twi"
LOCAL_DATA="/mnt/volume_d2wey28/projects/gloss-mt-2/data/en_tw_grand"
if [ -d "$LOCAL_DATA" ]; then
    DATA="$LOCAL_DATA"
else
    DATA="${DATA:-$HF_DATASET}"
fi

OUT="${OUT:-$DIR/checkpoints/umt5-base-en2tw}"
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

echo "=== Fine-tuning $MODEL ($DIRECTION) on Nkrane-MT entity-grounded dataset ($DATA) ==="
$PYTHON training/train_seq2seq.py \
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
