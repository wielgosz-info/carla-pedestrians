#!/usr/bin/env python3
"""
Compare OpenPose vs AlphaPose results
"""
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_openpose_results(csv_path):
    """Load OpenPose results from CSV"""
    df = pd.read_csv(csv_path)
    
    # Extract pose columns (assuming BODY_25 format with 25 keypoints)
    pose_cols = [col for col in df.columns if col.startswith('pose_')]
    
    results = []
    for _, row in df.iterrows():
        # Check if pose exists (not all zeros)
        pose_data = row[pose_cols].values
        has_pose = not np.all(pose_data == 0)
        
        if has_pose:
            # Calculate average confidence from every 3rd value (x, y, conf pattern)
            confidences = pose_data[2::3]  # Every 3rd value starting from index 2
            avg_conf = np.mean(confidences[confidences > 0]) if np.any(confidences > 0) else 0
            
            results.append({
                'frame_id': int(row['frame_id']),
                'pedestrian_id': int(row['pedestrian_id']),
                'avg_confidence': avg_conf,
                'num_keypoints': 25,
                'detected': True
            })
        else:
            results.append({
                'frame_id': int(row['frame_id']),
                'pedestrian_id': int(row['pedestrian_id']),
                'avg_confidence': 0,
                'num_keypoints': 25,
                'detected': False
            })
    
    return pd.DataFrame(results)

def load_alphapose_results(json_path):
    """Load AlphaPose results from JSON"""
    with open(json_path) as f:
        data = json.load(f)
    
    results = []
    for item in data:
        results.append({
            'frame_id': item['frame_id'],
            'pedestrian_id': item['pedestrian_id'],
            'avg_confidence': item['score'],
            'num_keypoints': 17,
            'detected': True
        })
    
    return pd.DataFrame(results)

def compare_results(openpose_df, alphapose_df):
    """Compare the two methods"""
    
    print("="*80)
    print("OPENPOSE vs ALPHAPOSE COMPARISON")
    print("="*80)
    
    # Overall statistics
    print("\n1. OVERALL DETECTION RATES:")
    print("-" * 40)
    openpose_detected = openpose_df['detected'].sum()
    openpose_total = len(openpose_df)
    alphapose_detected = len(alphapose_df)
    
    print(f"OpenPose:")
    print(f"  Total pedestrians: {openpose_total}")
    print(f"  Poses detected: {openpose_detected} ({openpose_detected/openpose_total*100:.1f}%)")
    print(f"  Failed detections: {openpose_total - openpose_detected}")
    
    print(f"\nAlphaPose:")
    print(f"  Total pedestrians: {alphapose_detected}")
    print(f"  Poses detected: {alphapose_detected} (100.0%)")
    print(f"  Note: Uses ground truth boxes, so 100% detection rate")
    
    # Confidence comparison on matched pedestrians
    print("\n2. CONFIDENCE SCORES (matched pedestrians only):")
    print("-" * 40)
    
    # Merge on frame_id and pedestrian_id
    merged = openpose_df[openpose_df['detected']].merge(
        alphapose_df,
        on=['frame_id', 'pedestrian_id'],
        suffixes=('_openpose', '_alphapose')
    )
    
    print(f"Matched detections: {len(merged)}")
    
    if len(merged) > 0:
        print(f"\nOpenPose confidence:")
        print(f"  Mean: {merged['avg_confidence_openpose'].mean():.3f}")
        print(f"  Median: {merged['avg_confidence_openpose'].median():.3f}")
        print(f"  Std: {merged['avg_confidence_openpose'].std():.3f}")
        
        print(f"\nAlphaPose confidence:")
        print(f"  Mean: {merged['avg_confidence_alphapose'].mean():.3f}")
        print(f"  Median: {merged['avg_confidence_alphapose'].median():.3f}")
        print(f"  Std: {merged['avg_confidence_alphapose'].std():.3f}")
        
        # Frames comparison
        print("\n3. PER-FRAME STATISTICS:")
        print("-" * 40)
        
        openpose_frames = openpose_df[openpose_df['detected']].groupby('frame_id').size()
        alphapose_frames = alphapose_df.groupby('frame_id').size()
        
        print(f"OpenPose:")
        print(f"  Frames with poses: {len(openpose_frames)}")
        print(f"  Avg detections per frame: {openpose_frames.mean():.1f}")
        
        print(f"\nAlphaPose:")
        print(f"  Frames with poses: {len(alphapose_frames)}")
        print(f"  Avg detections per frame: {alphapose_frames.mean():.1f}")
        
        # Create visualization
        print("\n4. CREATING COMPARISON PLOTS...")
        print("-" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Confidence distribution
        ax = axes[0, 0]
        ax.hist(merged['avg_confidence_openpose'], bins=30, alpha=0.5, label='OpenPose', color='blue')
        ax.hist(merged['avg_confidence_alphapose'], bins=30, alpha=0.5, label='AlphaPose', color='red')
        ax.set_xlabel('Average Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Scatter comparison
        ax = axes[0, 1]
        ax.scatter(merged['avg_confidence_openpose'], merged['avg_confidence_alphapose'], 
                   alpha=0.5, s=10)
        ax.plot([0, 1], [0, 1], 'r--', label='y=x')
        ax.set_xlabel('OpenPose Confidence')
        ax.set_ylabel('AlphaPose Confidence')
        ax.set_title('Confidence Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Detection rate by frame
        ax = axes[1, 0]
        frames = sorted(openpose_df['frame_id'].unique())[:50]
        op_rates = [openpose_df[openpose_df['frame_id']==f]['detected'].mean() for f in frames]
        ax.plot(frames, op_rates, label='OpenPose', marker='o', markersize=3)
        ax.axhline(y=1.0, color='r', linestyle='--', label='AlphaPose (100%)')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Detection Rate Over Frames')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Box plot comparison
        ax = axes[1, 1]
        data_to_plot = [merged['avg_confidence_openpose'], merged['avg_confidence_alphapose']]
        ax.boxplot(data_to_plot, labels=['OpenPose', 'AlphaPose'])
        ax.set_ylabel('Confidence Score')
        ax.set_title('Confidence Score Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/pose_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved comparison plot to /tmp/pose_comparison.png")
        
        # Correlation analysis
        print("\n5. CORRELATION ANALYSIS:")
        print("-" * 40)
        corr = merged['avg_confidence_openpose'].corr(merged['avg_confidence_alphapose'])
        print(f"Pearson correlation: {corr:.3f}")
        
        # Which method has higher confidence?
        higher_alpha = (merged['avg_confidence_alphapose'] > merged['avg_confidence_openpose']).sum()
        higher_open = (merged['avg_confidence_openpose'] > merged['avg_confidence_alphapose']).sum()
        
        print(f"\nConfidence comparison:")
        print(f"  AlphaPose higher: {higher_alpha} ({higher_alpha/len(merged)*100:.1f}%)")
        print(f"  OpenPose higher: {higher_open} ({higher_open/len(merged)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"✓ OpenPose: {openpose_detected}/{openpose_total} detections ({openpose_detected/openpose_total*100:.1f}%)")
    print(f"✓ AlphaPose: {alphapose_detected}/{alphapose_detected} detections (100.0%)")
    print(f"✓ Matched comparisons: {len(merged)}")
    print("="*80)

def main():
    openpose_csv = '/datasets/clear_noon/Town05/20251104-172725/labels_with_poses.csv'
    alphapose_json = '/tmp/alphapose_gt_results.json'
    
    print("Loading OpenPose results...")
    openpose_df = load_openpose_results(openpose_csv)
    
    print("Loading AlphaPose results...")
    alphapose_df = load_alphapose_results(alphapose_json)
    
    print("\nComparing results...")
    compare_results(openpose_df, alphapose_df)

if __name__ == '__main__':
    main()
