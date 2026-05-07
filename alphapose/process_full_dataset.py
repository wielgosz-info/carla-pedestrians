#!/usr/bin/env python3
"""
Process full PedSynth++ dataset with AlphaPose using ground truth boxes
"""
import os
import sys
import cv2
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, '/workspace/AlphaPose')

def load_pose_model():
    """Load AlphaPose model once"""
    from alphapose.models import builder
    from easydict import EasyDict as edict
    import yaml
    
    cfg_file = '/workspace/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml'
    checkpoint = '/workspace/AlphaPose/pretrained_models/fast_res50_256x192.pth'
    
    with open(cfg_file) as f:
        cfg = edict(yaml.safe_load(f))
    
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    pose_model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    pose_model = pose_model.cuda().eval()
    
    return pose_model

def process_person(model, img, bbox):
    """Extract pose for one person"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop person
    person_img = img[max(0,y1):y2, max(0,x1):x2]
    if person_img.size == 0:
        return None
    
    # Resize to 256x192
    inp = cv2.resize(person_img, (192, 256))
    inp = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
    inp = inp.unsqueeze(0).cuda()
    
    # Inference
    with torch.no_grad():
        hm = model(inp)
    
    # Get keypoints from heatmap
    hm = hm.cpu().numpy()[0]
    keypoints = []
    for i in range(17):  # COCO has 17 keypoints
        heatmap = hm[i]
        idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Scale back to original bbox
        x = x1 + (idx[1] / heatmap.shape[1]) * (x2 - x1)
        y = y1 + (idx[0] / heatmap.shape[0]) * (y2 - y1)
        conf = float(heatmap[idx])
        
        keypoints.extend([x, y, conf])
    
    return keypoints

def process_video_directory(model, video_dir, output_csv):
    """Process one video directory"""
    labels_csv = os.path.join(video_dir, 'labels.csv')
    
    if not os.path.exists(labels_csv):
        return None, "No labels.csv"
    
    # Load labels
    df = pd.read_csv(labels_csv)
    
    # Add pose columns (17 keypoints * 3 = 51 values)
    pose_cols = []
    for i in range(17):
        pose_cols.extend([f'pose_x_{i}', f'pose_y_{i}', f'pose_conf_{i}'])
    
    for col in pose_cols:
        df[col] = 0.0
    
    # Process frames
    processed = 0
    failed = 0
    
    for frame_num in df['frame_id'].unique():
        frame_data = df[df['frame_id'] == frame_num]
        
        # Load image
        img_path = os.path.join(video_dir, f'{frame_num:06d}.png')
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Process each pedestrian
        for idx, row in frame_data.iterrows():
            bbox = [row['bbox_x_min'], row['bbox_y_min'], 
                   row['bbox_x_max'], row['bbox_y_max']]
            
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                failed += 1
                continue
            
            keypoints = process_person(model, img, bbox)
            
            if keypoints:
                # Update dataframe with pose
                for i, val in enumerate(keypoints):
                    df.at[idx, pose_cols[i]] = val
                processed += 1
            else:
                failed += 1
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save results
    df.to_csv(output_csv, index=False)
    
    return processed, failed

def find_all_videos(base_dir):
    """Find all video directories with labels.csv"""
    videos = []
    
    for weather_dir in Path(base_dir).glob('*'):
        if not weather_dir.is_dir():
            continue
        
        for town_dir in weather_dir.glob('Town*'):
            if not town_dir.is_dir():
                continue
            
            for video_dir in town_dir.glob('*'):
                if not video_dir.is_dir():
                    continue
                
                labels_path = video_dir / 'labels.csv'
                if labels_path.exists():
                    videos.append({
                        'path': str(video_dir),
                        'weather': weather_dir.name,
                        'town': town_dir.name,
                        'video_id': video_dir.name,
                        'rel_path': str(video_dir.relative_to(base_dir))
                    })
    
    return videos

def main():
    base_dir = '/datasets'
    output_base = '/tmp/alphapose_results'
    checkpoint_file = '/tmp/alphapose_progress.json'
    
    print("="*80)
    print("ALPHAPOSE FULL DATASET PROCESSING")
    print("="*80)
    print(f"Input: {base_dir}")
    print(f"Output: {output_base}")
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Load checkpoint if exists
    processed_videos = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            processed_videos = set(checkpoint.get('processed', []))
        print(f"\nResuming from checkpoint: {len(processed_videos)} videos already processed")
    
    # Find all videos
    print("\nScanning dataset...")
    all_videos = find_all_videos(base_dir)
    print(f"Found {len(all_videos)} videos total")
    
    # Filter out already processed
    videos_to_process = [v for v in all_videos if v['path'] not in processed_videos]
    print(f"Videos to process: {len(videos_to_process)}")
    
    if len(videos_to_process) == 0:
        print("\nAll videos already processed!")
        return
    
    # Load model
    print("\nLoading AlphaPose model...")
    model = load_pose_model()
    print("Model loaded!")
    
    # Process videos
    print("\nProcessing videos...")
    stats = {
        'total_processed': 0,
        'total_failed': 0,
        'videos_completed': 0,
        'videos_failed': 0
    }
    
    start_time = time.time()
    
    for video_info in tqdm(videos_to_process, desc="Videos"):
        video_path = video_info['path']
        rel_path = video_info['rel_path']
        output_csv = os.path.join(output_base, rel_path, 'labels_with_alphapose.csv')
        
        try:
            result = process_video_directory(model, video_path, output_csv)
            
            if result[0] is not None:
                processed, failed = result
                stats['total_processed'] += processed
                stats['total_failed'] += failed
                stats['videos_completed'] += 1
                
                # Update checkpoint every 10 videos
                if stats['videos_completed'] % 10 == 0:
                    processed_videos.add(video_path)
                    with open(checkpoint_file, 'w') as f:
                        json.dump({'processed': list(processed_videos)}, f)
            else:
                stats['videos_failed'] += 1
                
        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            stats['videos_failed'] += 1
            continue
    
    # Final checkpoint
    processed_videos.update([v['path'] for v in videos_to_process[:stats['videos_completed']]])
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed': list(processed_videos)}, f)
    
    elapsed = time.time() - start_time
    
    # Final report
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"\nVideos completed: {stats['videos_completed']}")
    print(f"Videos failed: {stats['videos_failed']}")
    print(f"Total poses extracted: {stats['total_processed']}")
    print(f"Total failures: {stats['total_failed']}")
    if stats['total_processed'] + stats['total_failed'] > 0:
        print(f"Success rate: {stats['total_processed']/(stats['total_processed']+stats['total_failed'])*100:.1f}%")
    print(f"\nTotal time: {elapsed/3600:.2f} hours")
    if stats['videos_completed'] > 0:
        print(f"Average per video: {elapsed/stats['videos_completed']:.1f} seconds")
    print(f"\nResults saved to: {output_base}")
    print(f"Copy results with: docker cp alphapose_alphapose_1:/tmp/alphapose_results /data2/nriaz/")
    print("="*80)

if __name__ == '__main__':
    main()
