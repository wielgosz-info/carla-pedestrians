#!/usr/bin/env python3
"""
Combine all AlphaPose results into a single CSV file with poses as list in one column
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import ast

def find_all_alphapose_csvs(base_dir):
    """Find all labels_with_alphapose.csv files"""
    csv_files = []
    
    for csv_path in Path(base_dir).rglob('labels_with_alphapose.csv'):
        csv_files.append(csv_path)
    
    return sorted(csv_files)

def extract_pose_as_list(row):
    """Extract pose keypoints as a list of [x, y, conf] for each keypoint"""
    pose_data = []
    
    # AlphaPose has 17 keypoints
    for i in range(17):
        x = row.get(f'pose_x_{i}', 0.0)
        y = row.get(f'pose_y_{i}', 0.0)
        conf = row.get(f'pose_conf_{i}', 0.0)
        pose_data.append([float(x), float(y), float(conf)])
    
    return pose_data

def combine_csvs(csv_files, output_path):
    """Combine all CSV files into one with poses as list"""
    print(f"Found {len(csv_files)} CSV files to combine")
    
    if len(csv_files) == 0:
        print("ERROR: No CSV files found!")
        return
    
    all_rows = []
    total_rows = 0
    
    print("\nProcessing CSV files...")
    for csv_path in tqdm(csv_files, desc="Loading"):
        try:
            df = pd.read_csv(csv_path)
            
            # Get base columns (everything except pose columns)
            pose_cols = [col for col in df.columns if col.startswith('pose_')]
            base_cols = [col for col in df.columns if col not in pose_cols]
            
            # Process each row
            for _, row in df.iterrows():
                # Get base data
                row_data = {col: row[col] for col in base_cols}
                
                # Add pose as list
                row_data['pose_keypoints'] = extract_pose_as_list(row)
                
                all_rows.append(row_data)
                total_rows += 1
                
        except Exception as e:
            print(f"\nError loading {csv_path}: {e}")
            continue
    
    if len(all_rows) == 0:
        print("ERROR: No valid data loaded!")
        return
    
    print(f"\nCreating dataframe with {total_rows:,} rows...")
    combined_df = pd.DataFrame(all_rows)
    
    # Reorder columns to put pose_keypoints at the end
    base_cols = [col for col in combined_df.columns if col != 'pose_keypoints']
    combined_df = combined_df[base_cols + ['pose_keypoints']]
    
    print(f"Saving to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    
    return combined_df, file_size_mb

def analyze_combined_dataset(df):
    """Analyze the combined dataset"""
    print("\n" + "="*80)
    print("COMBINED ALPHAPOSE DATASET STATISTICS")
    print("="*80)
    
    # Basic stats
    print(f"\nTotal rows: {len(df):,}")
    print(f"Total unique pedestrians: {df['pedestrian_id'].nunique():,}")
    print(f"Total unique videos: {df['video_id'].nunique():,}")
    
    # Parse pose_keypoints to analyze
    print("\nPose quality statistics:")
    
    def count_valid_keypoints(pose_str):
        """Count keypoints with confidence > 0"""
        try:
            pose = ast.literal_eval(pose_str) if isinstance(pose_str, str) else pose_str
            return sum(1 for kp in pose if kp[2] > 0)
        except:
            return 0
    
    def avg_confidence(pose_str):
        """Calculate average confidence of valid keypoints"""
        try:
            pose = ast.literal_eval(pose_str) if isinstance(pose_str, str) else pose_str
            valid_confs = [kp[2] for kp in pose if kp[2] > 0]
            return sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
        except:
            return 0.0
    
    # Sample analysis on first 1000 rows for speed
    sample_df = df.head(min(1000, len(df)))
    
    print("  Analyzing sample of 1000 rows...")
    sample_df['num_valid_keypoints'] = sample_df['pose_keypoints'].apply(count_valid_keypoints)
    sample_df['avg_pose_conf'] = sample_df['pose_keypoints'].apply(avg_confidence)
    
    print(f"  Average keypoints detected: {sample_df['num_valid_keypoints'].mean():.1f} / 17")
    print(f"  Average confidence: {sample_df['avg_pose_conf'].mean():.3f}")
    print(f"  Median confidence: {sample_df['avg_pose_conf'].median():.3f}")
    
    # Quality distribution
    high_quality = (sample_df['num_valid_keypoints'] >= 15).sum()
    medium_quality = ((sample_df['num_valid_keypoints'] >= 10) & (sample_df['num_valid_keypoints'] < 15)).sum()
    low_quality = (sample_df['num_valid_keypoints'] < 10).sum()
    
    print(f"\nQuality distribution (sample):")
    print(f"  High quality (15+ keypoints): {high_quality} ({high_quality/len(sample_df)*100:.1f}%)")
    print(f"  Medium quality (10-14 keypoints): {medium_quality} ({medium_quality/len(sample_df)*100:.1f}%)")
    print(f"  Low quality (<10 keypoints): {low_quality} ({low_quality/len(sample_df)*100:.1f}%)")
    
    # Distance analysis (if available)
    if 'distance_to_ego' in df.columns:
        print("\nDistance distribution:")
        bins = [0, 10, 20, 30, 40, 100]
        labels = ['0-10m', '10-20m', '20-30m', '30-40m', '40m+']
        
        df['distance_bin'] = pd.cut(df['distance_to_ego'], bins=bins, labels=labels)
        dist_counts = df['distance_bin'].value_counts().sort_index()
        
        for bin_label in labels:
            if bin_label in dist_counts.index:
                count = dist_counts[bin_label]
                pct = count / len(df) * 100
                print(f"  {bin_label}: {count:,} ({pct:.1f}%)")
    
    # Crossing behavior (if available)
    if 'crossing' in df.columns:
        print("\nCrossing behavior:")
        crossing_counts = df['crossing'].value_counts()
        for behavior, count in crossing_counts.items():
            print(f"  {behavior}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Show example of pose format
    print("\n" + "="*80)
    print("EXAMPLE POSE FORMAT")
    print("="*80)
    print("\nFirst row pose_keypoints:")
    example_pose = df['pose_keypoints'].iloc[0]
    if isinstance(example_pose, str):
        example_pose = ast.literal_eval(example_pose)
    
    print(f"Format: List of 17 keypoints, each [x, y, confidence]")
    print(f"Sample (first 3 keypoints):")
    for i, kp in enumerate(example_pose[:3]):
        keypoint_names = ['Nose', 'Left Eye', 'Right Eye']
        print(f"  {keypoint_names[i]}: x={kp[0]:.2f}, y={kp[1]:.2f}, conf={kp[2]:.3f}")
    
    print("\n" + "="*80)

def main():
    # Paths
    input_base = '/tmp/alphapose_results'
    output_csv = '/tmp/combined_alphapose_dataset.csv'
    
    print("="*80)
    print("COMBINING ALPHAPOSE RESULTS")
    print("="*80)
    print(f"Input directory: {input_base}")
    print(f"Output file: {output_csv}")
    print(f"Pose format: Single column with [[x,y,c], [x,y,c], ...]")
    
    start_time = time.time()
    
    # Find all CSV files
    csv_files = find_all_alphapose_csvs(input_base)
    
    if len(csv_files) == 0:
        print(f"\nERROR: No CSV files found in {input_base}")
        print("Make sure AlphaPose processing completed successfully.")
        return
    
    # Combine
    result = combine_csvs(csv_files, output_csv)
    
    if result is None:
        return
    
    combined_df, file_size_mb = result
    
    elapsed = time.time() - start_time
    
    # Analyze
    analyze_combined_dataset(combined_df)
    
    # Final summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Output file: {output_csv}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Total time: {elapsed:.1f} seconds")
    print("\nColumns in output CSV:")
    print("  - All original columns (video_id, frame_id, pedestrian_id, bbox, etc.)")
    print("  - pose_keypoints: [[x1,y1,c1], [x2,y2,c2], ..., [x17,y17,c17]]")
    print("\nTo copy to host:")
    print(f"  docker cp alphapose_alphapose_1:{output_csv} ~/combined_alphapose_dataset.csv")
    print("="*80)

if __name__ == '__main__':
    main()
