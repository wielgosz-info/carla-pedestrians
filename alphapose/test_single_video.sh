#!/bin/bash

VIDEO="/datasets/clear_noon/Town05/20251104-172725/front_cam_30fps.mp4"
OUTPUT_DIR="/tmp/alphapose_test"

echo "Testing AlphaPose on: $VIDEO"
mkdir -p $OUTPUT_DIR

cd /workspace/AlphaPose

# Download correct AlphaPose model
if [ ! -f "pretrained_models/fast_res50_256x192.pth" ]; then
    echo "Downloading AlphaPose model..."
    wget --progress=bar:force https://github.com/MVIG-SJTU/AlphaPose/releases/download/v0.5.0/fast_res50_256x192.pth -P pretrained_models/
    ls -lh pretrained_models/
fi

# Download correct YOLO-SPP weights
if [ ! -f "detector/yolo/data/yolov3-spp.weights" ]; then
    echo "Downloading YOLO-SPP detector..."
    wget --progress=bar:force https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3-spp.weights -P detector/yolo/data/
    ls -lh detector/yolo/data/
fi

echo "Running AlphaPose..."
time python3 scripts/demo_inference.py \
    --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint pretrained_models/fast_res50_256x192.pth \
    --video "$VIDEO" \
    --outdir $OUTPUT_DIR \
    --save_video \
    --format coco

echo ""
echo "Done! Results:"
ls -lh $OUTPUT_DIR/
