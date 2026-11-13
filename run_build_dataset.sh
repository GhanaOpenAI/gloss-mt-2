#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
SRC_REPO="/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo"
VENV="$PROJ/.venv"

cd "$PROJ/repo"

echo "[$(date '+%F %T')] Building GLOSS-MT-2 bidirectional preserved datasets..."

$VENV/bin/python scripts/build_preserved_dataset.py \
    --data "$SRC_REPO/data" \
    --align-cache "$SRC_REPO/build/gold_alignment_cache.jsonl" \
    --out-dir "$PROJ/data" \
    --chunk-tokens 192

echo "[$(date '+%F %T')] Datasets built and saved in $PROJ/data!"
