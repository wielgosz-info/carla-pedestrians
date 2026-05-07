#!/usr/bin/env python3
"""
Create train/test/validation split stratified by weather condition
Split: 30 train / 10 test / 10 validation per weather (from ~50 videos per weather)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def find_all_videos_by_weather(base_dir):
    """Find all video directories organized by weather"""
    weather_videos = {}
    
    for weather_dir in Path(base_dir).glob('*'):
        if not weather_dir.is_dir():
            continue
        
        weather = weather_dir.name
        videos = []
        
        for town_dir in weather_dir.glob('Town*'):
            if not town_dir.is_dir():
                continue
            
            for video_dir in town_dir.glob('*'):
                if not video_dir.is_dir():
                    continue
                
                csv_path = video_dir / 'labels_with_alphapose.csv'
                if csv_path.exists():
                    # Store relative path from base for loading later
                    rel_path = video_dir.relative_to(base_dir)
                    videos.append({
                        'weather': weather,
                        'town': town_dir.name,
                        'video_id': video_dir.name,
                        'rel_path': str(rel_path),
                        'csv_path': str(csv_path)
                    })
        
        if videos:
            weather_videos[weather] = videos
    
    return weather_videos

def create_stratified_split(weather_videos, test_per_weather=10, val_per_weather=10, random_seed=42):
    """Create stratified train/test/val split"""
    np.random.seed(random_seed)
    
    print("\n" + "="*80)
    print("WEATHER DISTRIBUTION")
    print("="*80)
    for weather in sorted(weather_videos.keys()):
        print(f"  {weather}: {len(weather_videos[weather])} videos")
    
    # Create splits
    train_videos = []
    test_videos = []
    val_videos = []
    
    print("\n" + "="*80)
    print("CREATING SPLITS")
    print("="*80)
    
    for weather in sorted(weather_videos.keys()):
        videos = weather_videos[weather]
        num_videos = len(videos)
        
        # Shuffle videos for this weather
        shuffled = np.random.permutation(videos).tolist()
        
        # Determine split sizes
        actual_test = min(test_per_weather, num_videos)
        actual_val = min(val_per_weather, max(0, num_videos - actual_test))
        
        # Split
        test_vids = shuffled[:actual_test]
        val_vids = shuffled[actual_test:actual_test + actual_val]
        train_vids = shuffled[actual_test + actual_val:]
        
        test_videos.extend(test_vids)
        val_videos.extend(val_vids)
        train_videos.extend(train_vids)
        
        print(f"\n{weather}:")
        print(f"  Total: {num_videos} videos")
        print(f"  Train: {len(train_vids)} videos")
        print(f"  Test: {len(test_vids)} videos")
        print(f"  Val: {len(val_vids)} videos")
    
    return train_videos, test_videos, val_videos

def load_and_combine_csvs(video_list):
    """Load CSVs for given videos and combine"""
    all_dfs = []
    
    for video_info in video_list:
        try:
            df = pd.read_csv(video_info['csv_path'])
            # Add weather and relative path
            df['weather'] = video_info['weather']
            df['rel_path'] = video_info['rel_path']
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {video_info['csv_path']}: {e}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

def main():
    # Paths
    input_base = '/tmp/alphapose_results'
    output_dir = '/tmp/alphapose_splits'
    
    print("="*80)
    print("CREATING TRAIN/TEST/VAL SPLIT")
    print("="*80)
    print(f"Input directory: {input_base}")
    print(f"Output directory: {output_dir}")
    print(f"Split strategy: ~30 train / 10 test / 10 val per weather")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Find all videos organized by weather
    print("\nScanning dataset...")
    weather_videos = find_all_videos_by_weather(input_base)
    
    total_videos = sum(len(videos) for videos in weather_videos.values())
    print(f"Found {total_videos} videos across {len(weather_videos)} weather conditions")
    
    # Create splits
    train_vids, test_vids, val_vids = create_stratified_split(
        weather_videos,
        test_per_weather=10,
        val_per_weather=10,
        random_seed=42
    )
    
    print("\n" + "="*80)
    print("LOADING AND COMBINING CSVS")
    print("="*80)
    
    print(f"\nLoading train set ({len(train_vids)} videos)...")
    train_df = load_and_combine_csvs(train_vids)
    
    print(f"Loading test set ({len(test_vids)} videos)...")
    test_df = load_and_combine_csvs(test_vids)
    
    print(f"Loading validation set ({len(val_vids)} videos)...")
    val_df = load_and_combine_csvs(val_vids)
    
    print(f"\nDataframes created:")
    print(f"  Train: {len(train_df):,} rows from {len(train_vids)} videos")
    print(f"  Test: {len(test_df):,} rows from {len(test_vids)} videos")
    print(f"  Val: {len(val_df):,} rows from {len(val_vids)} videos")
    
    # Save splits
    print("\n" + "="*80)
    print("SAVING SPLITS")
    print("="*80)
    
    train_path = f"{output_dir}/train.csv"
    test_path = f"{output_dir}/test.csv"
    val_path = f"{output_dir}/val.csv"
    metadata_path = f"{output_dir}/split_metadata.json"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    # Save metadata
    metadata = {
        'train_videos': [{'weather': v['weather'], 'video_id': v['video_id'], 'rel_path': v['rel_path']} for v in train_vids],
        'test_videos': [{'weather': v['weather'], 'video_id': v['video_id'], 'rel_path': v['rel_path']} for v in test_vids],
        'val_videos': [{'weather': v['weather'], 'video_id': v['video_id'], 'rel_path': v['rel_path']} for v in val_vids],
        'weather_distribution': {weather: len(videos) for weather, videos in weather_videos.items()},
        'split_config': {
            'test_per_weather': 10,
            'val_per_weather': 10,
            'random_seed': 42
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Get file sizes
    train_size_mb = Path(train_path).stat().st_size / (1024 * 1024)
    test_size_mb = Path(test_path).stat().st_size / (1024 * 1024)
    val_size_mb = Path(val_path).stat().st_size / (1024 * 1024)
    
    # Final summary
    print("\n" + "="*80)
    print("SPLIT STATISTICS")
    print("="*80)
    
    for name, df in [('Train', train_df), ('Test', test_df), ('Validation', val_df)]:
        if len(df) == 0:
            print(f"\n{name} Set: EMPTY")
            continue
        
        print(f"\n{name} Set:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Unique videos: {df['video_id'].nunique()}")
        print(f"  Weather distribution:")
        for weather, count in df['weather'].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {weather}: {count:,} rows ({pct:.1f}%)")
    
    print("\n" + "="*80)
    print("OUTPUT SUMMARY")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  Train: {train_path} ({train_size_mb:.1f} MB, {len(train_df):,} rows)")
    print(f"  Test: {test_path} ({test_size_mb:.1f} MB, {len(test_df):,} rows)")
    print(f"  Val: {val_path} ({val_size_mb:.1f} MB, {len(val_df):,} rows)")
    print(f"  Metadata: {metadata_path}")
    
    if len(train_df) + len(test_df) + len(val_df) > 0:
        total = len(train_df) + len(test_df) + len(val_df)
        print(f"\nSplit ratios:")
        print(f"  Train: {len(train_df)/total*100:.1f}%")
        print(f"  Test: {len(test_df)/total*100:.1f}%")
        print(f"  Val: {len(val_df)/total*100:.1f}%")
    
    print("\nTo copy to host:")
    print(f"  docker cp alphapose_alphapose_1:{output_dir} ~/alphapose_splits/")
    print("="*80)

if __name__ == '__main__':
    main()
