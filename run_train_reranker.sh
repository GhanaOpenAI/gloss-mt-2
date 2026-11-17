#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

MODEL="Davlan/afro-xlmr-base"
OUT="${OUT:-$DIR/checkpoints/reranker-afro-xlmr}"
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

# 1. Build reranker dataset if not present
RERANKER_DATA="$DIR/data/reranker_dataset"
if [ -d "/mnt/volume_d2wey28/projects/gloss-mt-2/data/reranker_dataset" ]; then
    RERANKER_DATA="/mnt/volume_d2wey28/projects/gloss-mt-2/data/reranker_dataset"
fi

if [ ! -d "$RERANKER_DATA" ]; then
    echo "=== Building Cross-Encoder Reranker dataset ==="
    $PYTHON scripts/build_reranker_dataset.py \
        --source-dataset "ghanaopenai/nkrane-mt-parallel-corpus-eng-twi" \
        --dict-path "nkrane_mt/data/twi_dictionary.tsv.gz" \
        --out-dir "$RERANKER_DATA"
fi

# 2. Train Afro-XLMR Cross-Encoder Reranker
echo "=== Training Afro-XLMR Cross-Encoder Reranker on $RERANKER_DATA ==="
$PYTHON training/train_reranker.py \
    --model "$MODEL" \
    --data "$RERANKER_DATA" \
    --out "$OUT" \
    --epochs 2 \
    --batch-size 16 \
    --grad-accum 4 \
    --lr 1e-5 \
    --max-length 160 \
    --eval-steps 500 \
    --early-stopping-patience 10 \
    --push "ghanaopenai/nkrane-mt-reranker-afro-xlmr" \
    >> "$LOG_DIR/train_reranker.log" 2>&1

echo "=== Afro-XLMR Reranker training complete! Best model saved to $OUT/final ==="
