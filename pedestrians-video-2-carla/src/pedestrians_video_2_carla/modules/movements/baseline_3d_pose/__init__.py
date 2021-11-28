"""
This module contains models based on the [PyTorch implementation](https://github.com/weigq/3d_pose_baseline_pytorch)
of 3D pose baseline from the following paper:

```bibtex
@inproceedings{martinez2017simple,
title={A simple yet effective baseline for 3d human pose estimation},
author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
booktitle={ICCV},
year={2017}
}
```
"""

from .baseline_3d_pose import Baseline3DPose
from .baseline_3d_pose_rot import Baseline3DPoseRot
