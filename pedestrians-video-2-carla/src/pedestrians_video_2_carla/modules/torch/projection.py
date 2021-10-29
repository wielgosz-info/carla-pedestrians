from typing import Dict, Iterator, Tuple, Union

import numpy as np
from pedestrians_video_2_carla.renderers import MergingMethod
from pedestrians_video_2_carla.renderers.carla_renderer import CarlaRenderer
from pedestrians_video_2_carla.renderers.points_renderer import PointsRenderer
from pedestrians_video_2_carla.renderers.renderer import Renderer
from pedestrians_video_2_carla.renderers.source_renderer import SourceRenderer
from pedestrians_video_2_carla.skeletons.points import MAPPINGS
from pedestrians_video_2_carla.skeletons.points.carla import CARLA_SKELETON
from pedestrians_video_2_carla.skeletons.points.openpose import BODY_25, COCO
from pedestrians_video_2_carla.transforms.hips_neck import (
    CarlaHipsNeckNormalize, HipsNeckDeNormalize)
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
                 input_nodes: Union[BODY_25, COCO] = BODY_25,
                 output_nodes: CARLA_SKELETON = CARLA_SKELETON,
                 projection_transform=None,
                 max_videos=1,
                 fps=30.0,
                 merging_method: MergingMethod = MergingMethod.square,
                 enabled_renderers: Dict[str, bool] = None,
                 data_dir=None,
                 **kwargs
                 ) -> None:
        super().__init__()

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        if projection_transform is None:
            projection_transform = CarlaHipsNeckNormalize(input_nodes=output_nodes)
        self.projection_transform = projection_transform

        # set on every batch
        self.__pedestrians = None
        self.__pose_projection = None
        self.__world_locations = None
        self.__world_rotations = None
        self.__denormalizer = None

        # renderers
        if enabled_renderers is None:
            enabled_renderers = {
                'source': False,
                'input': True,
                'projection': True,
                'carla': False
            }
        self.__enabled_renderers = enabled_renderers

        if sum(enabled_renderers.values()) < 3 and merging_method == MergingMethod.square:
            merging_method = MergingMethod.horizontal
        self.__merging_method = merging_method

        self.__max_videos = max_videos
        self.__fps = fps
        self.__zeros_renderer = Renderer(max_videos=self.__max_videos)
        self.__source_renderer = SourceRenderer(
            data_dir=data_dir,
            max_videos=max_videos
        ) if self.__enabled_renderers['source'] else None
        self.__input_renderer = PointsRenderer(
            input_nodes=self.input_nodes,
            max_videos=max_videos
        ) if self.__enabled_renderers['input'] else None
        self.__projection_renderer = PointsRenderer(
            input_nodes=self.projection_transform.extractor.input_nodes,
            max_videos=max_videos
        ) if self.__enabled_renderers['projection'] else None
        self.__carla_renderer = CarlaRenderer(
            fps=self.__fps,
            max_videos=max_videos
        ) if self.__enabled_renderers['carla'] else None

    def on_batch_start(self, batch, batch_idx, dataloader_idx):
        (frames, ages, genders, *_) = batch
        batch_size = len(frames)

        # create pedestrian object for each clip in batch
        self.__pedestrians = [
            ControlledPedestrian(world=None, age=age, gender=gender,
                                 pose_cls=P3dPose, device=frames.device)
            for (age, gender) in zip(ages, genders)
        ]
        # only create one - we're assuming that camera is setup in the same for way for each pedestrian
        self.__pose_projection = P3dPoseProjection(
            device=frames.device, pedestrian=self.__pedestrians[0])
        # TODO: handle initial world transform matching instead of setting all zeros?

        self.__world_locations = torch.zeros(
            (batch_size, 3), device=frames.device)
        self.__world_rotations = torch.eye(3, device=frames.device).reshape(
            (1, 3, 3)).repeat((batch_size, 1, 1))

        movements = torch.zeros(
            (batch_size, 1, len(self.output_nodes), 3), device=frames.device)
        (relative_loc, relative_rot) = zip(*[
            p.current_pose.tensors
            for p in self.__pedestrians
        ])
        # prepare individual denormalization for each pedestrian
        # since there can be different reference poses depending on
        # adult/child male/female
        self.__denormalizer = HipsNeckDeNormalize().from_projection(
            self.projection_transform.extractor,
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

    def project_pose(self, pose_change_batch: Tensor, world_loc_change_batch: Tensor, world_rot_change_batch: Tensor):
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

    def forward(self, pose_inputs, targets, world_loc_inputs=None, world_rot_inputs=None):
        if world_loc_inputs is None:
            world_loc_inputs = torch.zeros((*pose_inputs.shape[:2], 3),
                                           device=pose_inputs.device)  # no world loc change

        if world_rot_inputs is None:
            world_rot_inputs = torch.zeros((*pose_inputs.shape[:2], 3),
                                           device=pose_inputs.device)  # no world rot change

        projected_pose = self.project_pose(
            pose_inputs,
            world_loc_inputs,
            world_rot_inputs,
        )
        normalized_projection = self.projection_transform(projected_pose)

        if type(self.input_nodes) == CARLA_SKELETON:
            carla_indices = slice(None)
            openpose_indices = slice(None)
        else:
            mappings = MAPPINGS[self.input_nodes]
            (carla_indices, openpose_indices) = zip(
                *[(c.value, o.value) for (c, o) in mappings])

        common_projection = normalized_projection[..., carla_indices, 0:2]
        common_openpose = targets[..., openpose_indices, 0:2]

        return (common_openpose, common_projection, projected_pose, normalized_projection)

    def render(self,
               batch: Tensor,
               projected_pose: Tensor,
               pose_change: Tensor,
               stage: str
               ) -> Iterator[Tuple[Tensor, Tuple[str, str, int]]]:
        """
        Prepares video data. **It doesn't save anything!**

        :param batch: As fed into .forward
        :type batch: Tensor
        :param projected_pose: Output of the projection layer
        :type projected_pose: Tensor
        :param pose_change: Output from the .forward
        :type pose_change: Tensor
        :param stage: train/val/test/predict
        :type stage: str
        :return: List of videos and (potential) name parts
        :rtype: Tuple[List[Tensor], Tuple[str]]
        """

        (frames, ages, genders, video_ids, pedestrian_ids, clip_ids, frame_ids) = batch
        image_size = self.__pose_projection.image_size

        denormalized_frames = self.__denormalizer(frames)

        source_videos = None
        if self.__source_renderer is not None:
            source_videos = self.__source_renderer.render(
                video_ids, pedestrian_ids, clip_ids, frame_ids, stage, image_size)

        input_videos = None
        if self.__input_renderer is not None:
            input_videos = self.__input_renderer.render(
                denormalized_frames, image_size)

        projection_videos = None
        if self.__projection_renderer is not None:
            projection_videos = self.__projection_renderer.render(
                projected_pose, image_size)

        carla_videos = None
        if self.__carla_renderer is not None:
            carla_videos = self.__carla_renderer.render(
                pose_change, ages, genders, image_size
            )

        if self.__merging_method == MergingMethod.square:
            source_videos = self.__zeros_renderer.render(
                frames, image_size) if source_videos is None else source_videos
            input_videos = self.__zeros_renderer.render(
                frames, image_size) if input_videos is None else input_videos
            projection_videos = self.__zeros_renderer.render(
                frames, image_size) if projection_videos is None else projection_videos
            carla_videos = self.__zeros_renderer.render(
                frames, image_size) if carla_videos is None else carla_videos
        else:
            rendered_videos = min(self.__max_videos, len(frames))
            source_videos = [None] * \
                rendered_videos if source_videos is None else source_videos
            input_videos = [None] * \
                rendered_videos if input_videos is None else input_videos
            projection_videos = [
                None] * rendered_videos if projection_videos is None else projection_videos
            carla_videos = [None] * \
                rendered_videos if carla_videos is None else carla_videos

        name_parts = zip(video_ids, pedestrian_ids, clip_ids)

        for (input_vid, projection_vid, carla_vid, source_vid, parts) in zip(input_videos, projection_videos, carla_videos, source_videos, name_parts):
            if self.__merging_method.value < 2:
                merged_vid = torch.tensor(np.concatenate(
                    [a for a in (source_vid, input_vid, projection_vid,
                                 carla_vid) if a is not None],
                    axis=self.__merging_method.value+1
                ))
            else:  # square
                merged_vid = torch.tensor(
                    np.concatenate((
                        np.concatenate((source_vid, input_vid), axis=2),
                        np.concatenate((carla_vid, projection_vid), axis=2)
                    ), axis=1)
                )
            yield merged_vid, parts
