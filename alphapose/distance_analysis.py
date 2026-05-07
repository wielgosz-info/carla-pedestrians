import pandas as pd
import json
import numpy as np

print("Loading data...")
openpose = pd.read_csv('/datasets/clear_noon/Town05/20251104-172725/labels_with_poses.csv')
with open('/tmp/alphapose_gt_results.json') as f:
    alphapose = json.load(f)

alpha_df = pd.DataFrame(alphapose)

# Merge
merged = openpose.merge(alpha_df[['frame_id', 'pedestrian_id', 'score']], 
                        on=['frame_id', 'pedestrian_id'], 
                        how='inner',
                        suffixes=('', '_alpha'))

print("="*80)
print("DISTANCE ANALYSIS - AlphaPose Test Set (First 50 Frames)")
print("="*80)

print(f"\nTotal matched detections: {len(merged)}")
print(f"Mean distance: {merged['distance_to_ego'].mean():.2f}m")
print(f"Median distance: {merged['distance_to_ego'].median():.2f}m")
print(f"Min distance: {merged['distance_to_ego'].min():.2f}m")
print(f"Max distance: {merged['distance_to_ego'].max():.2f}m")

# Distance bins
print("\n" + "="*80)
print("DISTRIBUTION BY DISTANCE")
print("="*80)
bins = [0, 10, 20, 30, 40, 100]
labels = ['0-10m', '10-20m', '20-30m', '30-40m', '40+m']

for i in range(len(bins)-1):
    mask = ((merged['distance_to_ego'] >= bins[i]) & 
            (merged['distance_to_ego'] < bins[i+1]))
    subset = merged[mask]
    count = len(subset)
    pct = count/len(merged)*100
    
    if count > 0:
        # Calculate OpenPose confidence for this bin
        pose_cols = [col for col in openpose.columns if col.startswith('pose_')]
        confidences = []
        for _, row in subset.iterrows():
            pose_data = row[pose_cols].values
            confs = pose_data[2::3]  # Every 3rd value
            avg_conf = confs[confs > 0].mean() if (confs > 0).any() else 0
            confidences.append(avg_conf)
        
        op_conf = np.mean(confidences)
        alpha_conf = subset['score'].mean()
        
        print(f"\n{labels[i]}: {count} detections ({pct:.1f}%)")
        print(f"  OpenPose confidence: {op_conf:.3f}")
        print(f"  AlphaPose confidence: {alpha_conf:.3f}")
        print(f"  AlphaPose advantage: {alpha_conf - op_conf:.3f}")

# Overall comparison
print("\n" + "="*80)
print("OVERALL CONFIDENCE COMPARISON")
print("="*80)

# Calculate OpenPose confidence for all matched
pose_cols = [col for col in openpose.columns if col.startswith('pose_')]
all_op_confs = []
for _, row in merged.iterrows():
    pose_data = row[pose_cols].values
    confs = pose_data[2::3]
    avg_conf = confs[confs > 0].mean() if (confs > 0).any() else 0
    all_op_confs.append(avg_conf)

print(f"\nOpenPose (on matched set):")
print(f"  Mean: {np.mean(all_op_confs):.3f}")
print(f"  Median: {np.median(all_op_confs):.3f}")
print(f"  Std: {np.std(all_op_confs):.3f}")

print(f"\nAlphaPose (on matched set):")
print(f"  Mean: {merged['score'].mean():.3f}")
print(f"  Median: {merged['score'].median():.3f}")
print(f"  Std: {merged['score'].std():.3f}")

print(f"\nConfidence improvement: +{(merged['score'].mean() - np.mean(all_op_confs)):.3f}")
print("="*80)
