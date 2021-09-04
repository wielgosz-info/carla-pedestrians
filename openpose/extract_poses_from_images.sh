#!/bin/bash

cd /openpose # it needs to be run there to find 'models' dir

echo "Processing ${1}..."
./build/examples/openpose/openpose.bin \
    --image_dir ${1} \
    --write_json ${2} \
    --model_pose BODY_25 \
    --display 0 \
    --render_pose 1 \
    --disable_blending \
    --write_images ${2}
    # --hand