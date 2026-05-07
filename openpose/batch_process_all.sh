#!/bin/bash

# Batch process all PedSynth++ videos with OpenPose
DATASET_ROOT="/datasets"
LOG_FILE="/tmp/batch_processing.log"
PROGRESS_FILE="/tmp/batch_progress.txt"

# Weather conditions to process
WEATHER_CONDITIONS="clear_noon clear_sunset cloudy_noon dawn foggy_noon heavy_rain_noon night_clear night_foggy night_rainy rainy_sunset wet_noon"

echo "=========================================" | tee -a $LOG_FILE
echo "PedSynth++ Batch Pose Extraction" | tee -a $LOG_FILE
echo "Started: $(date)" | tee -a $LOG_FILE
echo "=========================================" | tee -a $LOG_FILE

total_videos=0
processed=0
skipped=0
failed=0

# Count total videos first
for weather in $WEATHER_CONDITIONS; do
    for video_dir in $DATASET_ROOT/$weather/Town*/*/; do
        [ -f "$video_dir/front_cam_30fps.mp4" ] && total_videos=$((total_videos + 1))
    done
done

echo "Total videos found: $total_videos" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Process each video
for weather in $WEATHER_CONDITIONS; do
    echo "Processing weather: $weather" | tee -a $LOG_FILE
    
    for video_dir in $DATASET_ROOT/$weather/Town*/*/; do
        [ -d "$video_dir" ] || continue
        [ -f "$video_dir/front_cam_30fps.mp4" ] || continue
        
        video_name=$(basename "$video_dir")
        output_csv="/tmp/${video_name}_labels_with_poses.csv"
        
        # Skip if already processed
        if [ -f "$output_csv" ]; then
            echo "  SKIP: $video_name (already done)" | tee -a $LOG_FILE
            skipped=$((skipped + 1))
            continue
        fi
        
        echo "  Processing: $video_name" | tee -a $LOG_FILE
        start_time=$(date +%s)
        
        # Create temporary pose output directory
        pose_output="/tmp/poses_${video_name}"
        mkdir -p "$pose_output"
        
        # Run OpenPose from /openpose directory (CRITICAL!)
        cd /openpose && ./build/examples/openpose/openpose.bin \
            --video "$video_dir/front_cam_30fps.mp4" \
            --write_json "$pose_output" \
            --model_pose BODY_25 \
            --display 0 \
            --render_pose 0 \
            --number_people_max 20 \
            >> $LOG_FILE 2>&1
        
        if [ $? -eq 0 ]; then
            # Merge poses with labels
            python3 /app/merge_poses_final.py "$video_dir" "$pose_output" >> $LOG_FILE 2>&1
            
            if [ $? -eq 0 ]; then
                processed=$((processed + 1))
                end_time=$(date +%s)
                duration=$((end_time - start_time))
                
                echo "    ✓ Success (${duration}s)" | tee -a $LOG_FILE
                
                # Update progress
                echo "$processed/$total_videos" > $PROGRESS_FILE
            else
                echo "    ✗ Merge failed" | tee -a $LOG_FILE
                failed=$((failed + 1))
            fi
        else
            echo "    ✗ OpenPose failed" | tee -a $LOG_FILE
            failed=$((failed + 1))
        fi
        
        # Clean up pose JSONs to save disk space
        rm -rf "$pose_output"
        
        # Progress update
        completed=$((processed + skipped + failed))
        echo "  Progress: $completed/$total_videos (Processed: $processed, Skipped: $skipped, Failed: $failed)" | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE
    done
done

echo "=========================================" | tee -a $LOG_FILE
echo "BATCH PROCESSING COMPLETE!" | tee -a $LOG_FILE
echo "Finished: $(date)" | tee -a $LOG_FILE
echo "Total videos: $total_videos" | tee -a $LOG_FILE
echo "Successfully processed: $processed" | tee -a $LOG_FILE
echo "Skipped (already done): $skipped" | tee -a $LOG_FILE
echo "Failed: $failed" | tee -a $LOG_FILE
echo "Results saved in: /tmp/*_labels_with_poses.csv" | tee -a $LOG_FILE
echo "=========================================" | tee -a $LOG_FILE
