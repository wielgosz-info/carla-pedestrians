from typing import Union

from pedestrians_video_2_carla.skeletons.nodes import MAPPINGS
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.openpose import (
    BODY_25_SKELETON, COCO_SKELETON)
from pedestrians_video_2_carla.transforms.hips_neck import (
    CarlaHipsNeckExtractor, HipsNeckNormalize)
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from pedestrians_video_2_carla.walker_control.torch.pose_projection import \
    P3dPoseProjection
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix

import torch
from torch import nn
from torch.functional import Tensor


class ProjectionModule(nn.Module):
    def __init__(self,
                 input_nodes: Union[BODY_25_SKELETON, COCO_SKELETON,
                                    CARLA_SKELETON] = BODY_25_SKELETON,
                 output_nodes: CARLA_SKELETON = CARLA_SKELETON,
                 projection_transform=None,
                 **kwargs
                 ) -> None:
        super().__init__()

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        if projection_transform is None:
            projection_transform = HipsNeckNormalize(
                CarlaHipsNeckExtractor(input_nodes=output_nodes))
        self.projection_transform = projection_transform

        # set on every batch
        self.__pedestrians = None
        self.__pose_projection = None
        self.__world_locations = None
        self.__world_rotations = None

    def on_batch_start(self, batch, batch_idx, dataloader_idx):
        (frames, _, meta) = batch
        batch_size = len(frames)

        # create pedestrian object for each clip in batch
        self.__pedestrians = [
            ControlledPedestrian(world=None, age=meta['age'][idx], gender=meta['gender'][idx],
                                 pose_cls=P3dPose, device=frames.device)
            for idx in range(batch_size)
        ]
        # only create one - we're assuming that camera is setup in the same for way for each pedestrian
        self.__pose_projection = P3dPoseProjection(
            device=frames.device, pedestrian=self.__pedestrians[0])

        # TODO: handle initial world transform matching instead of setting all zeros?
        self.__world_locations = torch.zeros(
            (batch_size, 3), device=frames.device)
        self.__world_rotations = torch.eye(3, device=frames.device).reshape(
            (1, 3, 3)).repeat((batch_size, 1, 1))

    def project_pose(self, pose_change_batch: Tensor, world_loc_change_batch: Tensor = None, world_rot_change_batch: Tensor = None) -> Tensor:
        """
        Handles calculation of the pose projection.

        :param pose_change_batch: (N - batch_size, L - clip_length, B - bones, 3 - rotations as euler angles in radians)
        :type pose_change_batch: Tensor
        :param world_loc_change_batch: (N - batch_size, L - clip_length, 3 - location changes)
        :type world_loc_change_batch: Tensor
        :param world_rot_change_batch: (N - batch_size, L - clip_length, 3 - rotation changes as euler angles in radians)
        :type world_rot_change_batch: Tensor
        :raises RuntimeError: when pose_change_batch dimensionality is incorrect
        :return: [description]
        :rtype: [type]
        """
        (batch_size, clip_length, points, features) = pose_change_batch.shape

        # TODO: switch batch and clip length dimensions?
        if pose_change_batch.ndim < 4:
            raise RuntimeError(
                'Pose changes should have shape of (N - batch_size, L - clip_length, B - bones, 3 - rotations as euler angles)')

        if world_loc_change_batch is None:
            world_loc_change_batch = torch.zeros((*pose_change_batch.shape[:2], 3),
                                                 device=pose_change_batch.device)  # no world loc change

        if world_rot_change_batch is None:
            world_rot_change_batch = torch.zeros((*pose_change_batch.shape[:2], 3),
                                                 device=pose_change_batch.device)  # no world rot change

        (prev_relative_loc, prev_relative_rot) = zip(*[
            p.current_pose.tensors
            for p in self.__pedestrians
        ])

        prev_relative_loc = torch.stack(prev_relative_loc)
        prev_relative_rot = torch.stack(prev_relative_rot)

        # get subsequent poses calculated
        # naively for now
        # TODO: wouldn't it be better if P3dPose and P3PoseProjection were directly sequence-aware?
        # so that we only get in the initial loc/rot and a sequence of changes
        absolute_loc = torch.empty(
            (batch_size, clip_length, points, features), device=pose_change_batch.device)
        pose: P3dPose = self.__pedestrians[0].current_pose
        for i in range(clip_length):
            (absolute_loc[:, i], _, prev_relative_rot) = pose.forward(
                pose_change_batch[:, i], prev_relative_loc, prev_relative_rot)

        world_rot_matrix_change_batch = euler_angles_to_matrix(
            world_rot_change_batch, "XYZ")
        projections = torch.empty_like(absolute_loc)
        for i in range(clip_length):
            self.__world_rotations = torch.bmm(
                self.__world_rotations,
                world_rot_matrix_change_batch[:, i]
            )
            self.__world_locations += world_loc_change_batch[:, i]
            projections[:, i] = self.__pose_projection.forward(
                absolute_loc[:, i],
                self.__world_locations,
                self.__world_rotations
            )
        return projections.reshape((batch_size, clip_length, points, 3))

    def forward(self, pose_inputs, targets, world_loc_inputs=None, world_rot_inputs=None):
        projected_pose = self.project_pose(
            pose_inputs,
            world_loc_inputs,
            world_rot_inputs,
        )
        normalized_projection = self.projection_transform(projected_pose)

        if self.input_nodes == CARLA_SKELETON:
            carla_indices = slice(None)
            input_indices = slice(None)
        else:
            mappings = MAPPINGS[self.input_nodes]
            (carla_indices, input_indices) = zip(
                *[(c.value, o.value) for (c, o) in mappings])

        common_projection = normalized_projection[..., carla_indices, 0:2]
        common_input = targets[..., input_indices, 0:2]

        return (common_input, common_projection, projected_pose, normalized_projection)
