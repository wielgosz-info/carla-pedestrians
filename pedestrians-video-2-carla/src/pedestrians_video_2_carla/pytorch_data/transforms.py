from enum import Enum
from typing import Union, Any

import torch
from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO
from torch.functional import Tensor

from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON


class HipsNeckNormalize(object):
    """
    Normalize each sample so that hips x,y = 0,0 and distance between hips & neck == 1.
    """

    def __init__(self, points: Enum) -> None:
        self.points = points

    def __call__(self, sample: Tensor, *args: Any, **kwds: Any) -> Any:
        hips = self._get_hips_point(sample)
        neck = self._get_neck_point(sample)
        dist = torch.linalg.vector_norm(neck - hips, dim=-1)

        sample[..., 0:2] = (sample[..., 0:2] -
                            torch.unsqueeze(hips, -2)) / dist[(..., ) + (None, ) * 2]

        return torch.nan_to_num(sample)

    def _get_hips_point(self, sample: Tensor):
        raise NotImplementedError()

    def _get_neck_point(self, sample: Tensor):
        raise NotImplementedError()


class OpenPoseHipsNeckNormalize(HipsNeckNormalize):
    """
    Normalize each sample containing OpenPose keypoints so that hips x,y = 0,0 and distance between hips & neck == 1.
    """

    def __init__(self, points: Union[BODY_25, COCO] = BODY_25) -> None:
        super().__init__(points)

    def _get_hips_point(self, sample: Tensor):
        try:
            return sample[..., self.points.hips__C.value, 0:2]
        except AttributeError:
            # since COCO does not have hips point, we're using mean of tights
            return sample[..., [self.points.thigh__L.value, self.points.thigh__R.value], 0:2].mean(axis=-2)

    def _get_neck_point(self, sample: Tensor):
        return sample[..., self.points.neck__C.value, 0:2]


class CarlaHipsNeckNormalize(HipsNeckNormalize):
    """
    Normalize each sample containing CARLA skeleton so that hips x,y = 0,0 and distance between hips & neck == 1.
    """

    def __init__(self, points: CARLA_SKELETON = CARLA_SKELETON) -> None:
        super().__init__(points)

    def _get_hips_point(self, sample: Tensor):
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between tights as a reference instead
        return sample[..., [self.points.crl_thigh__L.value, self.points.crl_thigh__R.value], 0:2].mean(axis=-2)

    def _get_neck_point(self, sample: Tensor):
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between shoulders as a reference instead
        return sample[..., [self.points.crl_shoulder__L.value, self.points.crl_shoulder__R.value], 0:2].mean(axis=-2)
