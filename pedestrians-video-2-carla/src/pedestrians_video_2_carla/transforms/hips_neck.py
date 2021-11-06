from enum import Enum
from typing import Any, Callable, Type, Union

import torch
from torch.functional import Tensor
from pedestrians_video_2_carla.skeletons.nodes import Skeleton

from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.openpose import BODY_25_SKELETON, COCO_SKELETON


class HipsNeckExtractor(object):
    def __init__(self, input_nodes: Type[Skeleton]) -> None:
        self.input_nodes = input_nodes

    def get_hips_point(self, sample: Tensor) -> Tensor:
        raise NotImplementedError()

    def get_neck_point(self, sample: Tensor) -> Tensor:
        raise NotImplementedError()


# TODO: should specific extractor be here or in e.g. skeletons.nodes.openpose?
class OpenPoseHipsNeckExtractor(HipsNeckExtractor):
    def __init__(self, input_nodes: Type[Union[BODY_25_SKELETON, COCO_SKELETON]] = BODY_25_SKELETON) -> None:
        super().__init__(input_nodes)

    def get_hips_point(self, sample: Tensor) -> Tensor:
        try:
            return sample[..., self.input_nodes.hips__C.value, :]
        except AttributeError:
            # since COCO does not have hips point, we're using mean of tights
            return sample[..., [self.input_nodes.thigh__L.value, self.input_nodes.thigh__R.value], :].mean(axis=-2)

    def get_neck_point(self, sample: Tensor) -> Tensor:
        return sample[..., self.input_nodes.neck__C.value, :]


# TODO: should specific extractor be here or in e.g. skeletons.nodes.carla?
class CarlaHipsNeckExtractor(HipsNeckExtractor):
    def __init__(self, input_nodes: Type[CARLA_SKELETON] = CARLA_SKELETON) -> None:
        super().__init__(input_nodes)

    def get_hips_point(self, sample: Tensor) -> Tensor:
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between tights as a reference instead
        return sample[..., [self.input_nodes.crl_thigh__L.value, self.input_nodes.crl_thigh__R.value], :].mean(axis=-2)

    def get_neck_point(self, sample: Tensor) -> Tensor:
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between shoulders as a reference instead
        return sample[..., [self.input_nodes.crl_shoulder__L.value, self.input_nodes.crl_shoulder__R.value], :].mean(axis=-2)


class HipsNeckNormalize(object):
    """
    Normalize each sample so that hips x,y = 0,0 and distance between hips & neck == 1.
    """

    def __init__(self, extractor: HipsNeckExtractor, near_zero: float = 1e-5) -> None:
        self.extractor = extractor
        self.__near_zero = near_zero

    def __call__(self, sample: Tensor, dim=2, *args: Any, **kwargs: Any) -> Tensor:
        hips = self.extractor.get_hips_point(sample)[..., 0:dim]
        neck = self.extractor.get_neck_point(sample)[..., 0:dim]
        dist = torch.linalg.vector_norm(neck - hips, dim=-1)

        normalized_sample = torch.empty_like(sample)
        normalized_sample[..., 0:dim] = (sample[..., 0:dim] -
                                         torch.unsqueeze(hips, -2)) / dist[(..., ) + (None, ) * 2]

        if dim == 2:
            normalized_sample[..., 2] = sample[..., 2]

        normalized_sample = torch.nan_to_num(
            normalized_sample, nan=0, posinf=0, neginf=0)

        # if confidence is 0, we will assume the point overlaps with hips
        # so that values that were originally 0,0 (not detected)
        # do not skew the values range
        if dim == 2:
            normalized_sample[..., 0:2] = normalized_sample[..., 0:2].where(
                normalized_sample[..., 2:] >= self.__near_zero, torch.tensor(0.0, device=normalized_sample.device))

        return normalized_sample


class HipsNeckDeNormalize(object):
    """
    Denormalize each sample based on distance between hips & neck and hips position.
    """

    def __call__(self, sample: Tensor, dist: Tensor, hips: Tensor, dim=2, *args: Any, **kwargs: Any) -> Tensor:
        denormalized_sample = torch.empty_like(sample)
        denormalized_sample[..., 0:dim] = (
            sample[..., 0:dim] * dist[(..., ) + (None, ) * 2]) + torch.unsqueeze(hips, -2)

        if dim == 2:
            denormalized_sample[..., 2] = sample[..., 2]

        return denormalized_sample

    def from_projection(self, extractor: HipsNeckExtractor, projected_pose: Tensor) -> Callable:
        hips = extractor.get_hips_point(projected_pose)
        neck = extractor.get_neck_point(projected_pose)
        dist = torch.linalg.vector_norm(neck - hips, dim=-1)

        return lambda sample, dim=2: self(sample, dist[..., 0:dim], hips[..., 0:dim], dim)
