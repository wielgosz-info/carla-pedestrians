import pandas as pd
import json

print("Loading OpenPose CSV...")
openpose = pd.read_csv('/datasets/clear_noon/Town05/20251104-172725/labels_with_poses.csv')
print(f"OpenPose rows: {len(openpose)}")
print(f"OpenPose columns: {list(openpose.columns)[:10]}")

print("\nLoading AlphaPose JSON...")
with open('/tmp/alphapose_gt_results.json') as f:
    alphapose = json.load(f)
print(f"AlphaPose entries: {len(alphapose)}")

# Create AlphaPose dataframe
alpha_df = pd.DataFrame(alphapose)
print(f"AlphaPose DataFrame shape: {alpha_df.shape}")
print(f"AlphaPose columns: {list(alpha_df.columns)}")

# Check sample IDs
print(f"\nOpenPose sample frame_ids: {openpose['frame_id'].unique()[:5]}")
print(f"AlphaPose sample frame_ids: {alpha_df['frame_id'].unique()[:5]}")

print(f"\nOpenPose sample ped_ids: {openpose['pedestrian_id'].unique()[:10]}")
print(f"AlphaPose sample ped_ids: {alpha_df['pedestrian_id'].unique()[:10]}")

# Try merge
print("\nAttempting merge...")
merged = openpose.merge(alpha_df[['frame_id', 'pedestrian_id']], 
                        on=['frame_id', 'pedestrian_id'], 
                        how='inner')
print(f"Merged rows: {len(merged)}")

if len(merged) > 0:
    print("\nDistance analysis:")
    print(f"Mean distance: {merged['distance_to_ego'].mean():.2f}m")
    print(f"Median distance: {merged['distance_to_ego'].median():.2f}m")
