from typing import Any, Callable, Type

import torch
from torch.functional import Tensor


class HipsNeckExtractor(object):
    def __init__(self, input_nodes: Type['Skeleton']) -> None:
        self.input_nodes = input_nodes

    def get_hips_point(self, sample: Tensor) -> Tensor:
        raise NotImplementedError()

    def get_neck_point(self, sample: Tensor) -> Tensor:
        raise NotImplementedError()


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
        dist = torch.linalg.norm(neck - hips, dim=-1, ord=2)

        normalized_sample = torch.empty_like(sample)
        normalized_sample[..., 0:dim] = (sample[..., 0:dim] -
                                         torch.unsqueeze(hips, -2)) / dist[(..., ) + (None, ) * 2]

        if dim == 2:
            normalized_sample[..., 2] = sample[..., 2]

        if getattr(torch, 'nan_to_num', False):
            normalized_sample = torch.nan_to_num(
                normalized_sample, nan=0, posinf=0, neginf=0)
        else:
            normalized_sample = torch.where(torch.isnan(
                normalized_sample), torch.tensor(0.0, device=normalized_sample.device), normalized_sample)
            normalized_sample = torch.where(torch.isinf(
                normalized_sample), torch.tensor(0.0, device=normalized_sample.device), normalized_sample)

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
        dist = torch.linalg.norm(neck - hips, dim=-1, ord=2)

        return lambda sample, dim=2: self(sample, dist[..., 0:dim], hips[..., 0:dim], dim)
