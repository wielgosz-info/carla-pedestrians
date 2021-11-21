from typing import Any, Dict, List

import torch
from pedestrians_video_2_carla.skeletons.nodes.carla import CarlaHipsNeckExtractor
from pedestrians_video_2_carla.transforms.hips_neck import (HipsNeckDeNormalize, HipsNeckExtractor,
                                                            HipsNeckNormalize)
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from pedestrians_video_2_carla.walker_control.torch.pose_projection import \
    P3dPoseProjection
from torch.functional import Tensor


class ReferenceSkeletonsDenormalize(object):
    """
    Denormalizes (after optional autonormalization) the "absolute" skeleton
    coordinates by using reference skeletons.
    """

    def __init__(self,
                 autonormalize: bool = False,
                 extractor: HipsNeckExtractor = None
                 ) -> None:
        if extractor is None:
            extractor = CarlaHipsNeckExtractor()
        self._extractor = extractor

        if autonormalize:
            self.autonormalize = HipsNeckNormalize(self._extractor)
        else:
            self.autonormalize = lambda x, *args, **kwargs: x

        self.types = [
            ('adult', 'female'),
            ('adult', 'male'),
            ('child', 'female'),
            ('child', 'male')
        ]

        self.__reference_pedestrians = None
        self.__reference_abs = None
        self.__reference_projections = None

    def _get_reference_pedestrians(self, device):
        if self.__reference_pedestrians is None:
            self.__reference_pedestrians = [
                ControlledPedestrian(age=age, gender=gender,
                                     pose_cls=P3dPose, device=device)
                for (age, gender) in self.types
            ]
        return self.__reference_pedestrians

    def _get_reference_abs(self, device):
        if self.__reference_abs is None:
            pedestrians = self._get_reference_pedestrians(device)

            movements = torch.eye(3, device=device).reshape(
                (1, 1, 3, 3)).repeat((len(pedestrians), len(pedestrians[0].current_pose.empty), 1, 1))
            (relative_loc, relative_rot) = zip(*[
                p.current_pose.tensors
                for p in pedestrians
            ])

            self.__reference_abs = pedestrians[0].current_pose.forward(
                movements,
                torch.stack(relative_loc),
                torch.stack(relative_rot)
            )[0]
        return self.__reference_abs

    def _get_reference_projections(self, device):
        if self.__reference_projections is None:
            reference_abs = self._get_reference_abs(device)
            pedestrians = self._get_reference_pedestrians(device)

            # TODO: get camera settings from LitBaseMapper.projection
            pose_projection = P3dPoseProjection(
                device=device, pedestrian=pedestrians[0])

            # TODO: we're assuming no in-world movement for now!
            world_locations = torch.zeros(
                (len(reference_abs), 3), device=device)
            world_rotations = torch.eye(3, device=device).reshape(
                (1, 3, 3)).repeat((len(reference_abs), 1, 1))

            self.__reference_projections = pose_projection.forward(
                reference_abs,
                world_locations,
                world_rotations
            )
        return self.__reference_projections

    def get_projections(self, device) -> Dict[str, Tensor]:
        """
        Returns the reference projections as a dictionary.
        """
        return {
            t: p.unsqueeze(0) for t, p in zip(self.types, self._get_reference_projections(device))
        }

    def get_abs(self, device) -> Dict[str, Tensor]:
        """
        Returns the reference abs as a dictionary.
        """
        return {
            t: p.unsqueeze(0) for t, p in zip(self.types, self._get_reference_abs(device))
        }

    def from_projection(self, frames: Tensor, meta: Dict[str, List[Any]]) -> Tensor:
        frames = self.autonormalize(frames, dim=2)

        reference_projections = self.get_projections(frames.device)

        frame_projections = torch.stack([
            reference_projections[(age, gender)]
            for (age, gender) in zip(meta['age'], meta['gender'])
        ], dim=0)

        return HipsNeckDeNormalize().from_projection(self._extractor, frame_projections)(frames, dim=2)

    def from_abs(self, frames: Tensor, meta: Dict[str, List[Any]]) -> Tensor:
        frames = self.autonormalize(frames, dim=3)

        reference_abs = self.get_abs(frames.device)

        frame_abs = torch.stack([
            reference_abs[(age, gender)]
            for (age, gender) in zip(meta['age'], meta['gender'])
        ], dim=0)

        return HipsNeckDeNormalize().from_projection(self._extractor, frame_abs)(frames, dim=3)
