#!/usr/bin/env python3
"""
AlphaPose with ground truth boxes from CARLA
"""
import os
import cv2
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# AlphaPose imports
import sys
sys.path.insert(0, '/workspace/AlphaPose')

def load_pose_model():
    """Load AlphaPose model"""
    from alphapose.models import builder
    from alphapose.utils.config import update_config
    from easydict import EasyDict as edict
    import yaml
    
    cfg_file = '/workspace/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml'
    checkpoint = '/workspace/AlphaPose/pretrained_models/fast_res50_256x192.pth'
    
    # Load config
    with open(cfg_file) as f:
        cfg = edict(yaml.safe_load(f))
    
    # Build model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    pose_model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    pose_model = pose_model.cuda().eval()
    
    return pose_model, cfg

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

def main():
    frames_dir = '/datasets/clear_noon/Town05/20251104-172725'
    labels_csv = os.path.join(frames_dir, 'labels.csv')
    output_json = '/tmp/alphapose_gt_results.json'
    
    print("Loading AlphaPose model...")
    model, cfg = load_pose_model()
    
    print(f"Loading labels from {labels_csv}...")
    df = pd.read_csv(labels_csv)
    
    print(f"Processing {len(df)} pedestrians...")
    
    results = []
    
    # Group by frame_id
    for frame_num in tqdm(df['frame_id'].unique()[:50]):  # Test on first 50 frames
        frame_data = df[df['frame_id'] == frame_num]
        
        # Load image
        img_path = os.path.join(frames_dir, f'{frame_num:06d}.png')
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Process each pedestrian
        for _, row in frame_data.iterrows():
            bbox = [row['bbox_x_min'], row['bbox_y_min'], 
                   row['bbox_x_max'], row['bbox_y_max']]
            
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            
            keypoints = process_person(model, img, bbox)
            
            if keypoints:
                results.append({
                    'frame_id': int(frame_num),
                    'pedestrian_id': int(row['pedestrian_id']),
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'score': sum(keypoints[2::3]) / 17  # Average confidence
                })
    
    print(f"\nSaving {len(results)} results to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDone! Results summary:")
    print(f"  Total detections: {len(results)}")
    if len(results) > 0:
        avg_score = sum(r['score'] for r in results) / len(results)
        print(f"  Average confidence: {avg_score:.3f}")

if __name__ == '__main__':
    main()
