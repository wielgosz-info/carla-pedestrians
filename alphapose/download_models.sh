#!/bin/bash
cd /workspace/AlphaPose

# Try different model URLs
echo "Trying to download AlphaPose models..."

# Option 1: OneDrive link (from AlphaPose docs)
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn' -O pretrained_models/fast_res50_256x192.pth

# If that fails, try GoogleDrive
if [ ! -f "pretrained_models/fast_res50_256x192.pth" ] || [ ! -s "pretrained_models/fast_res50_256x192.pth" ]; then
    echo "Trying alternative download..."
    pip3 install gdown
    gdown 1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn -O pretrained_models/fast_res50_256x192.pth
fi

# Download YOLO
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3-spp.weights -P detector/yolo/data/

echo "Models downloaded:"
ls -lh pretrained_models/
ls -lh detector/yolo/data/
