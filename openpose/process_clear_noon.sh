#!/bin/bash

DATASET_ROOT="/datasets"
OUTPUT_ROOT="/tmp/PedSynth++_poses"
OPENPOSE_DIR="/openpose"
WEATHER="clear_noon"

echo "Starting pose extraction for ${WEATHER}..."
echo "Dataset: ${DATASET_ROOT}/${WEATHER}"
echo "Output: ${OUTPUT_ROOT}/${WEATHER}"

# Create output directory
mkdir -p ${OUTPUT_ROOT}/${WEATHER}

cd ${OPENPOSE_DIR}

echo "Counting videos..."
video_count=0
for video_file in ${DATASET_ROOT}/${WEATHER}/**/*.mp4; do
    if [ -f "${video_file}" ]; then
        ((video_count++))
    fi
done

echo "Found ${video_count} videos to process"
echo "======================================"

processed=0
for video_file in ${DATASET_ROOT}/${WEATHER}/**/*.mp4; do
    if [ ! -f "${video_file}" ]; then
        continue
    fi

    # Get relative path structure
    rel_path=${video_file#${DATASET_ROOT}/${WEATHER}/}
    video_dir=$(dirname "${rel_path}")
    video_name=$(basename "${video_file}" .mp4)

    # Create output directory
    output_dir="${OUTPUT_ROOT}/${WEATHER}/${video_dir}/${video_name}"
    mkdir -p "${output_dir}"

    echo "[${processed}/${video_count}] Processing: ${rel_path}"

    # Run OpenPose
    ${OPENPOSE_DIR}/build/examples/openpose/openpose.bin \
        --video "${video_file}" \
        --write_json "${output_dir}" \
        --model_pose BODY_25 \
        --display 0 \
        --render_pose 0 \
        --number_people_max 20 \
        > "${output_dir}/openpose_log.txt" 2>&1

    if [ $? -eq 0 ]; then
        echo "  [SUCCESS]"
    else
        echo "  [FAILED] Check: ${output_dir}/openpose_log.txt"
    fi

    ((processed++))
done

echo "======================================"
echo "All processing complete!"
echo "Processed ${processed}/${video_count} videos"
echo "Results saved to: ${OUTPUT_ROOT}/${WEATHER}"
EOF