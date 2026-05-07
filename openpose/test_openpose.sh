#!/bin/bash
OPENPOSE_DIR="/openpose"
VIDEO_FILE="/datasets/clear_noon/Town01/20251104-164948/front_cam_30fps.mp4"
OUTPUT_DIR="/tmp/test_pose_output"

mkdir -p ${OUTPUT_DIR}
cd ${OPENPOSE_DIR}

./build/examples/openpose/openpose.bin \
    --video "${VIDEO_FILE}" \
    --write_json "${OUTPUT_DIR}" \
    --model_pose BODY_25 \
    --display 0 \
    --render_pose 0 \
    --number_people_max 20
