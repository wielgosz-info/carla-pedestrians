
from typing import List
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
import torch
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F
import pytorch_lightning as pl
from pedestrians_video_2_carla.pytorch_walker_control.pose_projection import P3dPoseProjection

from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian
from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose


class LitBaseMapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.__pedestrians = []

    def _on_batch_start(self, batch, batch_idx, dataloader_idx):
        # TODO: is it OK if all of those are set as class fields instead of being passed through during forward?

        (ages, genders, frames) = batch
        batch_size = len(frames)

        # create pedestrian object for each clip in batch
        self.__pedestrians = [
            ControlledPedestrian(world=None, age=age, gender=gender,
                                 pose_cls=P3dPose, device=self.device)
            for (age, gender) in zip(ages, genders)
        ]
        # only create one - we're assuming that camera is setup in the same for way for each pedestrian
        self.__pose_projection = P3dPoseProjection(
            device=self.device, pedestrian=self.__pedestrians[0])
        # TODO: handle initial world transform matching instead of setting all zeros?

        self.__world_locations = torch.zeros(
            (batch_size, 3), device=self.device)
        self.__world_rotations = torch.eye(3, device=self._device).reshape(
            (1, 3, 3)).repeat((batch_size, 1, 1))

    def pose_projection(self, pose_change_batch: Tensor, world_loc_change_batch: Tensor, world_rot_change_batch: Tensor):
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

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, batch_idx, 'test')

    def _step(self, batch, batch_idx, stage):
        raise NotImplementedError()
