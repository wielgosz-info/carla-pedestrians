#!/usr/bin/env python3
import json, pandas, numpy as np, sys
from pathlib import Path

def iou(b1,b2):
    xi1,yi1,xi2,yi2=max(b1[0],b2[0]),max(b1[1],b2[1]),min(b1[2],b2[2]),min(b1[3],b2[3])
    if xi2<=xi1 or yi2<=yi1: return 0
    i=(xi2-xi1)*(yi2-yi1); u=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-i
    return i/u if u>0 else 0

def bbox_kp(k,t=0.3):
    v=[(p[0],p[1]) for p in k if p[2]>t]
    return [min(p[0] for p in v),min(p[1] for p in v),max(p[0] for p in v),max(p[1] for p in v)] if len(v)>=3 else None

vdir,pdir=Path(sys.argv[1]),Path(sys.argv[2])
df=pandas.read_csv(vdir/'labels.csv')

# Add compact pose columns
df['pose_keypoints'] = ''  # Will store JSON string: [[x1,y1,c1],[x2,y2,c2],...]
df['pose_matched'] = False
df['pose_iou'] = 0.0

m=0
for i,r in df.iterrows():
    js=list(pdir.glob(f"*_{r['frame_id']:012d}_keypoints.json"))
    if not js: continue
    d=json.load(open(js[0]))
    if not d['people']: continue
    gt=[r['bbox_x_min'],r['bbox_y_min'],r['bbox_x_max'],r['bbox_y_max']]
    bi,bm=0.2,None
    for p in d['people']:
        k=np.array(p['pose_keypoints_2d']).reshape(-1,3)
        pb=bbox_kp(k.tolist())
        if pb:
            io=iou(gt,pb)
            if io>bi: bi,bm=io,p['pose_keypoints_2d']
    if bm:
        m+=1
        # Store as compact JSON: [[x,y,conf], [x,y,conf], ...]
        k=np.array(bm).reshape(-1,3).tolist()
        df.at[i,'pose_keypoints'] = json.dumps(k)
        df.at[i,'pose_matched'] = True
        df.at[i,'pose_iou'] = bi

o=Path(f'/tmp/{vdir.name}_labels_with_poses.csv')
df.to_csv(o,index=False)
print(f"Matched: {m}/{len(df)} ({m/len(df)*100:.1f}%)")
print(f"Saved: {o}")
print(f"\nFormat: pose_keypoints column contains JSON list of 25 keypoints")
print(f"Each keypoint: [x, y, confidence]")
