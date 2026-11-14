#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
VENV="$PROJ/.venv"

cd "$PROJ/repo"
[ -f /mnt/volume_d2wey28/projects/twi-translation-tool-calling/.env ] && set -a && . /mnt/volume_d2wey28/projects/twi-translation-tool-calling/.env && set +a

echo "[$(date '+%F %T')] Building Grand Multi-Pillar GLOSS-MT-2 Dataset across GHOK, Pristine, News, and JW..."

$VENV/bin/python scripts/build_grand_dataset.py \
    --out-dir "$PROJ/data/en_tw_grand" \
    --cache-path "$PROJ/build/grand_alignment_cache.jsonl" \
    --chunks-cache "$PROJ/build/grand_chunks_cache.json" \
    --push ghanaopenai/gloss-mt-2-en-tw-grand \
    --pristine-sample-size 450000 \
    --workers-per-key 3

echo "[$(date '+%F %T')] Grand dataset creation and Hugging Face publication complete!"
