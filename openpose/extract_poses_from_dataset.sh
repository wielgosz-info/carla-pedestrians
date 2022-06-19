#!/bin/bash

dataset='PIE'
limit=360

cd /openpose # it needs to be run there to find 'models' dir

mkdir -p  /outputs/${dataset}
i=1
for filename in /datasets/${dataset}/videos/**/*.mp4; do
    set_name=$(dirname "$filename")
    set_name=$(basename "$set_name")
    mkdir -p  /outputs/${dataset}/${set_name}
    
    name=$(basename "$filename" .mp4)
    echo "Processing ${set_name}/${name}..."
    ./build/examples/openpose/openpose.bin \
        --video ${filename} \
        --write_json /outputs/${dataset}/${set_name}/${name} \
        --model_pose BODY_25 \
        --display 0 \
        --render_pose 0
        # --hand
        # --write_video /outputs/${dataset}/${name}.avi
    ((i=i+1))
    if [ $i -gt $limit ]; then
        break
    fi
done
