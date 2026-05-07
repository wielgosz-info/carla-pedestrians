#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path
import sys

TMP_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp')
OUTPUT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/tmp/combined_pedsynth_with_poses.csv')

print("Combining all labels_with_poses.csv files...")
print(f"Reading from: {TMP_DIR}")

csv_files = list(TMP_DIR.glob('*_labels_with_poses.csv'))
print(f"Found {len(csv_files)} CSV files\n")

# Create zero pose for pedestrians without detection
ZERO_POSE = json.dumps([[0.0, 0.0, 0.0] for _ in range(25)])

all_data = []
for i, csv_file in enumerate(csv_files, 1):
    try:
        df = pd.read_csv(csv_file)
        
        # Ensure columns exist
        if 'pose_keypoints' not in df.columns:
            df['pose_keypoints'] = ''
        if 'pose_matched' not in df.columns:
            df['pose_matched'] = False
        if 'pose_iou' not in df.columns:
            df['pose_iou'] = 0.0
        
        # Fill empty poses with zeros (25 keypoints, each [0,0,0])
        df.loc[df['pose_keypoints'] == '', 'pose_keypoints'] = ZERO_POSE
        df.loc[df['pose_keypoints'].isna(), 'pose_keypoints'] = ZERO_POSE
        
        df['pose_matched'] = df['pose_matched'].fillna(False)
        df['pose_iou'] = df['pose_iou'].fillna(0.0)
        
        all_data.append(df)
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(csv_files)} files...")
            
    except Exception as e:
        print(f"  ERROR reading {csv_file.name}: {e}")

combined_df = pd.concat(all_data, ignore_index=True)

print(f"\n✓ Combined dataset statistics:")
print(f"  Total pedestrians: {len(combined_df):,}")
print(f"  With poses detected: {combined_df['pose_matched'].sum():,}")
print(f"  Without poses (filled with zeros): {(~combined_df['pose_matched']).sum():,}")
print(f"  Match rate: {combined_df['pose_matched'].sum() / len(combined_df) * 100:.1f}%")

with_pose = combined_df[combined_df['pose_matched'] == True]
without_pose = combined_df[combined_df['pose_matched'] == False]

print(f"\nDistance analysis:")
print(f"  With pose - Avg: {with_pose['distance_to_ego'].mean():.1f}m")
print(f"  Without pose - Avg: {without_pose['distance_to_ego'].mean():.1f}m")

combined_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved: {OUTPUT_FILE}")
print(f"  Size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
