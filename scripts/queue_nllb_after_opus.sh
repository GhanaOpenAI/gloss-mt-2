#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
LOG_DIR="$PROJ/logs"
TARGET_PID=2835711

mkdir -p "$LOG_DIR"
echo "[$(date '+%F %T')] Queued NLLB-600M training. Monitoring OPUS-MT PID $TARGET_PID..." >> "$LOG_DIR/nllb_queue.log"

# Wait until OPUS-MT process exits
while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date '+%F %T')] OPUS-MT process ($TARGET_PID) has terminated. Waiting 15s for CUDA cleanup..." >> "$LOG_DIR/nllb_queue.log"
sleep 15

echo "[$(date '+%F %T')] Launching NLLB-200 600M en->tw training on H200..." >> "$LOG_DIR/nllb_queue.log"
cd "$PROJ/repo"
./run_train_nllb_600m_en2tw.sh >> "$LOG_DIR/nllb_queue.log" 2>&1

echo "[$(date '+%F %T')] NLLB-200 600M training and upload complete!" >> "$LOG_DIR/nllb_queue.log"
