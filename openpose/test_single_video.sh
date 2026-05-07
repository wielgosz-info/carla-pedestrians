#!/bin/bash

# Test script to verify pose extraction on a single video
# Usage: ./test_single_video.sh [path_to_video_directory]

if [ $# -eq 0 ]; then
    echo "Usage: ./test_single_video.sh [path_to_video_directory]"
    echo "Example: ./test_single_video.sh /media/nriaz/data2/nriaz/PedSynth++/clear_noon/town1/20251104-112639"
    exit 1
fi

VIDEO_DIR=$1
VIDEO_FILE="${VIDEO_DIR}/front_cam_30fps.mp4"
LABELS_FILE="${VIDEO_DIR}/labels.csv"
OUTPUT_DIR="./test_pose_output"
OPENPOSE_DIR="/openpose"

echo "==================================="
echo "PedSynth++ Pose Extraction Test"
echo "==================================="
echo "Video directory: ${VIDEO_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Check if video exists
if [ ! -f "${VIDEO_FILE}" ]; then
    echo "[ERROR] Video file not found: ${VIDEO_FILE}"
    exit 1
fi

# Check if labels exist
if [ ! -f "${LABELS_FILE}" ]; then
    echo "[WARN] Labels file not found: ${LABELS_FILE}"
else
    echo "[INFO] Labels file found: ${LABELS_FILE}"
    # Show label statistics
    if command -v python3 &> /dev/null; then
        echo "[INFO] Label statistics:"
        python3 << EOF
import pandas as pd
df = pd.read_csv('${LABELS_FILE}')
print(f"  Total frames: {df['frame_id'].max() + 1}")
print(f"  Total pedestrian detections: {len(df)}")
print(f"  Unique pedestrians: {df['pedestrian_id'].nunique() if 'pedestrian_id' in df.columns else 'N/A'}")
EOF
    fi
fi

echo ""

# Check OpenPose installation
if [ ! -d "${OPENPOSE_DIR}" ]; then
    echo "[ERROR] OpenPose directory not found: ${OPENPOSE_DIR}"
    echo "Please install OpenPose or update OPENPOSE_DIR variable"
    exit 1
fi

if [ ! -f "${OPENPOSE_DIR}/build/examples/openpose/openpose.bin" ]; then
    echo "[ERROR] OpenPose binary not found"
    exit 1
fi

echo "[INFO] OpenPose found at: ${OPENPOSE_DIR}"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Change to OpenPose directory
cd ${OPENPOSE_DIR}

echo "[PROCESSING] Extracting poses from video..."
echo "This may take a few minutes depending on video length and hardware."
echo ""

# Run OpenPose
./build/examples/openpose/openpose.bin \
    --video ${VIDEO_FILE} \
    --write_json ${OUTPUT_DIR}/openpose_output \
    --model_pose BODY_25 \
    --display 0 \
    --render_pose 0 \
    --number_people_max 20

# Check results
cd - > /dev/null

if [ -d "${OUTPUT_DIR}/openpose_output" ]; then
    JSON_COUNT=$(ls -1 ${OUTPUT_DIR}/openpose_output/*.json 2>/dev/null | wc -l)
    echo ""
    echo "==================================="
    echo "[SUCCESS] Pose extraction complete!"
    echo "==================================="
    echo "Output directory: ${OUTPUT_DIR}/openpose_output"
    echo "JSON files generated: ${JSON_COUNT}"
    echo ""

    # Show sample detection
    if [ ${JSON_COUNT} -gt 0 ]; then
        SAMPLE_JSON=$(ls ${OUTPUT_DIR}/openpose_output/*.json | head -n 1)
        echo "[SAMPLE] First frame detection:"
        if command -v python3 &> /dev/null; then
            python3 << EOF
import json
with open('${SAMPLE_JSON}', 'r') as f:
    data = json.load(f)
print(f"  People detected: {len(data['people'])}")
if len(data['people']) > 0:
    pose = data['people'][0]['pose_keypoints_2d']
    # Check how many keypoints have confidence > 0.3
    import numpy as np
    keypoints = np.array(pose).reshape(-1, 3)
    valid = (keypoints[:, 2] > 0.3).sum()
    print(f"  Valid keypoints (conf>0.3): {valid}/25")
EOF
        else
            cat ${SAMPLE_JSON} | head -n 20
        fi
    fi

    echo ""
    echo "Next step: Run pose-label merging"
    echo "python3 merge_poses_with_labels.py --dataset_root /path/to/dataset --pose_root ${OUTPUT_DIR}/.."
else
    echo ""
    echo "[ERROR] Pose extraction failed!"
    echo "Check the OpenPose output above for errors."
fi