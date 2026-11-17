#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[$(date '+%F %T')] Starting Nkrane-MT Training Pipeline..."

# 1. Train Inference Noun Phrase Marker (to convergence)
echo "[$(date '+%F %T')] [1/2] Training Inference Noun Phrase Marker..."
./run_train_marker.sh

# 2. Train OPUS-MT en->tw (to convergence with live BLEU evals)
echo "[$(date '+%F %T')] [2/2] Training Helsinki Marian OPUS-MT en->tw..."
./run_train_opus_mt_en2tw.sh

echo "[$(date '+%F %T')] All Nkrane-MT training completed successfully!"
