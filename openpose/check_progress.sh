#!/bin/bash

LOG_FILE="/tmp/batch_processing.log"
PROGRESS_FILE="/tmp/batch_progress.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "Batch processing not started yet."
    exit 0
fi

echo "========================================="
echo "Batch Processing Status"
echo "========================================="

if [ -f "$PROGRESS_FILE" ]; then
    cat $PROGRESS_FILE
fi

echo ""
echo "Recent activity:"
tail -20 $LOG_FILE

echo ""
echo "CSV files created:"
ls -lh /tmp/*_labels_with_poses.csv 2>/dev/null | wc -l

echo ""
echo "To see full log: tail -f /tmp/batch_processing.log"
