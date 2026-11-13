#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
SRC_REPO="/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo"
VENV="$PROJ/.venv"

cd "$PROJ/repo"
[ -f /mnt/volume_d2wey28/projects/twi-translation-tool-calling/.env ] && set -a && . /mnt/volume_d2wey28/projects/twi-translation-tool-calling/.env && set +a

echo "[$(date '+%F %T')] Building GLOSS-MT-2 dataset (complete sentences <= 512 chars with full determiners)..."

$VENV/bin/python scripts/build_gloss_mt_2_dataset.py \
    --data-dir "$SRC_REPO/data" \
    --cache-path "$PROJ/build/gold_alignment_cache_512.jsonl" \
    --out-dir "$PROJ/data/en_tw_512" \
    --push ghanaopenai/gloss-mt-2-en-tw \
    --max-chars 512 \
    --workers 64

echo "[$(date '+%F %T')] Dataset generation and publication complete!"
