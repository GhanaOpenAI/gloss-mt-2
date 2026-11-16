#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
VENV="$PROJ/.venv"
MODEL="Davlan/afro-xlmr-base"
OUT="$PROJ/checkpoints/reranker-afro-xlmr"
LOG_DIR="$PROJ/logs"

mkdir -p "$LOG_DIR" "$OUT"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ/repo"

# 1. Build reranker dataset if not present
if [ ! -d "$PROJ/data/reranker_dataset" ]; then
    echo "[$(date '+%F %T')] Building Cross-Encoder Reranker dataset..."
    $VENV/bin/python scripts/build_reranker_dataset.py \
        --source-dataset "ghanaopenai/nkrane-mt-parallel-corpus-eng-twi" \
        --dict-path "nkrane_mt/data/twi_dictionary.tsv.gz" \
        --out-dir "$PROJ/data/reranker_dataset"
fi

# 2. Train Afro-XLMR Cross-Encoder Reranker
echo "[$(date '+%F %T')] Training Afro-XLMR Cross-Encoder Reranker for contextual lexical disambiguation..."
$VENV/bin/python training/train_reranker.py \
    --model "$MODEL" \
    --data "$PROJ/data/reranker_dataset" \
    --out "$OUT" \
    --epochs 2 \
    --batch-size 16 \
    --grad-accum 4 \
    --lr 1e-5 \
    --max-length 160 \
    --eval-steps 500 \
    --early-stopping-patience 10 \
    >> "$LOG_DIR/train_reranker.log" 2>&1

echo "[$(date '+%F %T')] Afro-XLMR Reranker training complete! Best model saved to $OUT/final"
