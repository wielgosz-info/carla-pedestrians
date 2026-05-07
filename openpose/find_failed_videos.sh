#!/bin/bash

echo "Finding failed videos..."
echo ""

# Get all video directories
all_videos=$(find /data2/nriaz/PedSynth++/clear_noon -type f -name "front_cam_30fps.mp4" -exec dirname {} \;)

# Get processed videos from container
docker exec carla-pedestrians_openpose_1 bash -c "ls /tmp/*_labels_with_poses.csv 2>/dev/null" > /tmp/processed_list.txt

echo "Failed videos:"
for video_dir in $all_videos; do
    video_name=$(basename "$video_dir")
    
    if ! grep -q "$video_name" /tmp/processed_list.txt; then
        echo "  - $video_name"
        echo "    Path: $video_dir"
    fi
done

rm /tmp/processed_list.txt
