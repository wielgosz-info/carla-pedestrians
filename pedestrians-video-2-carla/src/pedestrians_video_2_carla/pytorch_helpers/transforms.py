from enum import Enum
from typing import Any, Union

import torch
from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO
from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON
from torch.functional import Tensor


class HipsNeckNormalize(object):
    """
    Normalize each sample so that hips x,y = 0,0 and distance between hips & neck == 1.
    """

    def __init__(self, input_nodes: Enum, near_zero: float = 1e-5) -> None:
        self.input_nodes = input_nodes
        self.__near_zero = near_zero

    def __call__(self, sample: Tensor, *args: Any, **kwds: Any) -> Any:
        hips = self._get_hips_point(sample)
        neck = self._get_neck_point(sample)
        dist = torch.linalg.vector_norm(neck - hips, dim=-1)

        sample[..., 0:2] = (sample[..., 0:2] -
                            torch.unsqueeze(hips, -2)) / dist[(..., ) + (None, ) * 2]

        num_sample = torch.nan_to_num(sample, nan=0, posinf=0, neginf=0)

        # if confidence is 0, we will assume the point overlaps with hips
        # so that values that were originally 0,0 (not detected)
        # do not skew the values range
        num_sample[..., 0:2] = num_sample[..., 0:2].where(
            num_sample[..., 2:] >= self.__near_zero, torch.tensor(0.0, device=num_sample.device))

        return num_sample

    def _get_hips_point(self, sample: Tensor):
        raise NotImplementedError()

    def _get_neck_point(self, sample: Tensor):
        raise NotImplementedError()


class OpenPoseHipsNeckNormalize(HipsNeckNormalize):
    """
    Normalize each sample containing OpenPose keypoints so that hips x,y = 0,0 and distance between hips & neck == 1.
    """

    def __init__(self, input_nodes: Union[BODY_25, COCO] = BODY_25) -> None:
        super().__init__(input_nodes)

    def _get_hips_point(self, sample: Tensor):
        try:
            return sample[..., self.input_nodes.hips__C.value, 0:2]
        except AttributeError:
            # since COCO does not have hips point, we're using mean of tights
            return sample[..., [self.input_nodes.thigh__L.value, self.input_nodes.thigh__R.value], 0:2].mean(axis=-2)

    def _get_neck_point(self, sample: Tensor):
        return sample[..., self.input_nodes.neck__C.value, 0:2]


class CarlaHipsNeckNormalize(HipsNeckNormalize):
    """
    Normalize each sample containing CARLA skeleton so that hips x,y = 0,0 and distance between hips & neck == 1.
    """

    def __init__(self, input_nodes: CARLA_SKELETON = CARLA_SKELETON) -> None:
        super().__init__(input_nodes)

    def _get_hips_point(self, sample: Tensor):
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between tights as a reference instead
        return sample[..., [self.input_nodes.crl_thigh__L.value, self.input_nodes.crl_thigh__R.value], 0:2].mean(axis=-2)

    def _get_neck_point(self, sample: Tensor):
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between shoulders as a reference instead
        return sample[..., [self.input_nodes.crl_shoulder__L.value, self.input_nodes.crl_shoulder__R.value], 0:2].mean(axis=-2)
