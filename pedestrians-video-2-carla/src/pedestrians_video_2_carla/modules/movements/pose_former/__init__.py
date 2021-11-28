"""
This module contains models based on the [PoseFormer implementation](https://github.com/zczcwh/PoseFormer)
from the following paper:

```bibtex
@article{zheng2021poseformer,
title={3D Human Pose Estimation with Spatial and Temporal Transformers},
author={Zheng, Ce and Zhu, Sijie and Mendieta, Matias and Yang,
    Taojiannan and Chen, Chen and Ding, Zhengming},
journal={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
year={2021}
}
```
"""

from .pose_former import PoseFormer
