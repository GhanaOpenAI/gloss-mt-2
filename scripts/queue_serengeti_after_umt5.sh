#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/gloss-mt-2"
LOG_DIR="$PROJ/logs"
TARGET_PID=3339354

mkdir -p "$LOG_DIR"
echo "[$(date '+%F %T')] Queued Serengeti-E250 training. Monitoring uMT5-Base PID $TARGET_PID..." >> "$LOG_DIR/serengeti_queue.log"

# Wait until uMT5 process exits
while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date '+%F %T')] uMT5-Base process ($TARGET_PID) has terminated. Waiting 15s for CUDA cleanup..." >> "$LOG_DIR/serengeti_queue.log"
sleep 15

echo "[$(date '+%F %T')] Launching Serengeti-E250 training on H200..." >> "$LOG_DIR/serengeti_queue.log"
cd "$PROJ/repo"
./run_train_reranker_serengeti.sh >> "$LOG_DIR/serengeti_queue.log" 2>&1

echo "[$(date '+%F %T')] Serengeti-E250 training and upload complete!" >> "$LOG_DIR/serengeti_queue.log"
