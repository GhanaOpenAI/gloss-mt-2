#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
VENV="$PROJ/.venv"

cd "$PROJ/repo"
[ -f /mnt/volume_d2wey28/projects/twi-translation-tool-calling/.env ] && set -a && . /mnt/volume_d2wey28/projects/twi-translation-tool-calling/.env && set +a

echo "[$(date '+%F %T')] Building Nkrane-MT Grand Multi-Pillar Parallel Dataset across GHOK, Pristine, News, and JW..."

$VENV/bin/python scripts/build_grand_dataset.py \
    --out-dir "$PROJ/data/en_tw_grand" \
    --cache-path "$PROJ/build/grand_alignment_cache.jsonl" \
    --chunks-cache "$PROJ/build/grand_chunks_cache.json" \
    --push ghanaopenai/nkrane-mt-parallel-corpus-eng-twi \
    --pristine-sample-size 450000 \
    --workers-per-key 3

echo "[$(date '+%F %T')] Nkrane-MT dataset creation and Hugging Face publication complete!"
