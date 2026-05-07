#!/bin/bash

FRAMES_DIR="/datasets/clear_noon/Town05/20251104-172725"
OUTPUT_DIR="/tmp/alphapose_frames_test"

echo "Testing AlphaPose on existing PNG frames..."
echo "Frames directory: $FRAMES_DIR"

# Count PNG files
NUM_FRAMES=$(ls $FRAMES_DIR/*.png 2>/dev/null | wc -l)
echo "Found $NUM_FRAMES PNG frames"

if [ $NUM_FRAMES -eq 0 ]; then
    echo "ERROR: No PNG frames found!"
    exit 1
fi

mkdir -p $OUTPUT_DIR

cd /workspace/AlphaPose

# Run AlphaPose on the PNG frames directory
echo "Running AlphaPose..."
time python3 scripts/demo_inference.py \
    --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint pretrained_models/fast_res50_256x192.pth \
    --indir "$FRAMES_DIR" \
    --outdir "$OUTPUT_DIR" \
    --format coco

echo ""
echo "Done! Results:"
ls -lh $OUTPUT_DIR/

if [ -f "$OUTPUT_DIR/alphapose-results.json" ]; then
    echo ""
    echo "Analysis:"
    python3 -c "
import json
data = json.load(open('$OUTPUT_DIR/alphapose-results.json'))
print(f'Total poses detected: {len(data)}')
if len(data) > 0:
    print(f'Average confidence: {sum(d[\"score\"] for d in data)/len(data):.3f}')
    print(f'Sample detection: {data[0]}')
"
fi
