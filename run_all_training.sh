#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
cd "$PROJ"

echo "[$(date '+%F %T')] Starting GLOSS-MT-2 Training Queue (OPUS-MT + NLLB 600M)..."

# 1. Train OPUS-MT en->tw
echo "[$(date '+%F %T')] [1/2] Running OPUS-MT en->tw..."
./run_train_opus_mt_en2tw.sh

# 2. Train NLLB 600M en->tw
echo "[$(date '+%F %T')] [2/2] Running NLLB 600M en->tw..."
./run_train_nllb_600m_en2tw.sh

echo "[$(date '+%F %T')] All GLOSS-MT-2 training runs completed successfully!"
