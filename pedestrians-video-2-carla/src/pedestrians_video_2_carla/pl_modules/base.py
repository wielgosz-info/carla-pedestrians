
from typing import Tuple, Union
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pedestrians_video_2_carla.pytorch_helpers.lossess import ProjectionLoss
from pedestrians_video_2_carla.pytorch_helpers.transforms import \
    HipsNeckDeNormalize
from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.pytorch_walker_control.pose_projection import \
    P3dPoseProjection
from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO
from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from torch.functional import Tensor


class LitBaseMapper(pl.LightningModule):
    def __init__(self, input_nodes: Union[BODY_25, COCO] = BODY_25, **kwargs):
        super().__init__()
        self.__pedestrians = []
        self.criterion = ProjectionLoss(
            self.pose_projection,
            input_nodes,
            **kwargs
        )
        self.__denormalizer = None  # set on every batch

    def _on_batch_start(self, batch, batch_idx, dataloader_idx):
        (frames, ages, genders, *_) = batch
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

        movements = torch.zeros(
            (batch_size, 1, len(CARLA_SKELETON), 3), device=frames.device)
        (relative_loc, relative_rot) = zip(*[
            p.current_pose.tensors
            for p in self.__pedestrians
        ])
        # prepare individual denormalization for each pedestrian
        # since there can be different reference poses depending on
        # adult/child male/female
        self.__denormalizer = HipsNeckDeNormalize().from_projection(
            self.criterion.projection_transform.extractor,
            self.__pose_projection.forward(
                self.__pedestrians[0].current_pose.forward(
                    movements,
                    torch.stack(relative_loc),
                    torch.stack(relative_rot)
                )[0],
                self.__world_locations,
                self.__world_rotations
            ).unsqueeze(1)
        )

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
        return self._step(batch, batch_idx, 'test')

    def _step(self, batch, batch_idx, stage):
        (frames, *_) = batch

        pose_change = self.forward(frames.to(self.device))

        (loss, projected_pose, _) = self.criterion(
            pose_change,
            frames
        )

        self.log('{}_loss'.format(stage), loss)

        if stage != 'train':
            self._log_videos(projected_pose, batch, batch_idx, stage)

        return loss

    def _log_videos(self, projected_pose: Tensor, batch: Tuple, batch_idx: int, stage: str):
        if batch_idx > 0 or self.current_epoch % 5 > 0:
            # we only want to log from the first batch and every 5th epoch
            # TODO: make this configurable
            return

        # TODO: allow to specify how many videos from each batch should be created
        max_videos = 1
        videos_dir = os.path.join(self.logger.log_dir, 'videos')
        if not os.path.exists(videos_dir):
            os.mkdir(videos_dir)

        (frames, ages, genders, video_ids, pedestrian_ids, clip_ids) = batch

        tb = self.logger.experiment

        # de-normalize OpenPose & align it with projection
        openpose = self.__denormalizer(frames)[..., 0:2].round().int().cpu().numpy()

        projection = projected_pose[..., 0:2].round().int().cpu().numpy()
        image_size = self.__pose_projection.image_size

        openpose_keys = [k.name for k in self.criterion.input_nodes]
        projection_keys = [
            k.name for k in self.criterion.projection_transform.extractor.input_nodes]

        # for every clip in batch
        # for every sequence frame in clip
        # draw video frame
        # TODO: rewrite to parallelize
        # TODO: add cropped clip from dataset?
        # TODO: add rendering from CARLA
        # TODO: draw bones and not only dots?
        videos = []
        for clip_idx in range(min(max_videos, len(openpose))):
            video = []
            openpose_clip = openpose[clip_idx]
            projection_clip = projection[clip_idx]
            video_id = video_ids[clip_idx]
            pedestrian_id = pedestrian_ids[clip_idx]
            clip_id = clip_ids[clip_idx]
            for (openpose_frame, projection_frame) in zip(openpose_clip, projection_clip):
                openpose_canvas = np.zeros((image_size[1], image_size[0], 4), np.uint8)
                openpose_img = self.__pose_projection.draw_projection_points(
                    openpose_canvas, openpose_frame, openpose_keys)

                projection_canvas = np.zeros(
                    (image_size[1], image_size[0], 4), np.uint8)
                projection_img = self.__pose_projection.draw_projection_points(
                    projection_canvas, projection_frame, projection_keys)

                # align them next to each other vertically
                img = np.concatenate((openpose_img, projection_img), axis=0)
                # TODO: add thin white border to separate groups from each other?
                video.append(torch.tensor(img[..., 0:3]))  # H,W,C
            video = torch.stack(video)  # T,H,W,C
            torchvision.io.write_video(
                os.path.join(videos_dir,
                             '{}-{}-{:0>2d}-ep{:0>4d}.mp4'.format(
                                 video_id,
                                 pedestrian_id,
                                 clip_id,
                                 self.current_epoch
                             )),
                video,
                fps=30.0
            )
            videos.append(video)

        videos = torch.stack(videos).permute(0, 1, 4, 2, 3)  # B,T,H,W,C -> B,T,C,H,W

        tb.add_video('{}_gt_and_projection_points'.format(stage),
                     videos, self.current_epoch, fps=30.0)
