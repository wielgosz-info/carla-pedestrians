import copy
from queue import Empty, Queue
from typing import List, Tuple, Union

import carla
import numpy as np
import PIL
import torch
from pedestrians_video_2_carla.pytorch_helpers.transforms import (
    CarlaHipsNeckNormalize, HipsNeckDeNormalize)
from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.pytorch_walker_control.pose_projection import \
    P3dPoseProjection
from pedestrians_video_2_carla.utils.destroy import destroy
from pedestrians_video_2_carla.utils.openpose import (BODY_25, COCO,
                                                      COMMON_NODES)
from pedestrians_video_2_carla.utils.setup import (setup_camera,
                                                   setup_client_and_world)
from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from torch import nn
from torch.functional import Tensor


class ProjectionModule(nn.Module):
    def __init__(self,
                 input_nodes: Union[BODY_25, COCO] = BODY_25,
                 output_nodes: CARLA_SKELETON = CARLA_SKELETON,
                 projection_transform=None,
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

        if type(self.input_nodes) == COCO:
            mappings = COMMON_NODES['CARLA_2_COCO']
        else:  # default
            mappings = COMMON_NODES['CARLA_2_BODY_25']

        (carla_indices, openpose_indices) = zip(
            *[(c.value, o.value) for (c, o) in mappings])

        common_projection = normalized_projection[..., carla_indices, 0:2]
        common_openpose = targets[..., openpose_indices, 0:2]

        return (common_openpose, common_projection, projected_pose, normalized_projection)

    def render(self, batch: Tensor, projected_pose: Tensor, pose_change: Tensor, max_videos: int = 1, fps: float = 30.0) -> Tuple[List[Tensor], Tuple[str]]:
        """
        Prepares video data. **It doesn't save anything!**

        :param batch: [description]
        :type batch: Tensor
        :param projected_pose: [description]
        :type projected_pose: Tensor
        :param pose_change: [description]
        :type pose_change: Tensor
        :param max_videos: [description], defaults to 1
        :type max_videos: int, optional
        :param fps: [description], defaults to 30.0
        :type fps: float, optional
        :return: [description]
        :rtype: Tuple[List[Tensor], Tuple[str]]
        """
        return ([], [])  # disable all rendering for now

        # TODO: optimize & refactor
        (frames, ages, genders, video_ids, pedestrian_ids, clip_ids) = batch

        concatenation_axis = 1  # 0-vertically, 1-horizontally, TODO: 2-square
        rendered_videos = min(max_videos, len(frames))

        # de-normalize OpenPose & align it with projection
        openpose = self.__denormalizer(frames)[..., 0:2].round().int().cpu().numpy()

        projection = projected_pose[..., 0:2].round().int().cpu().numpy()
        image_size = self.__pose_projection.image_size

        openpose_keys = [k.name for k in self.input_nodes]
        projection_keys = [
            k.name for k in self.projection_transform.extractor.input_nodes]

        # prepare connection to carla as needed - TODO: should this be in (logging) epoch start?
        client, world = setup_client_and_world(fps=fps)

        # for every clip in batch
        # for every sequence frame in clip
        # draw video frame
        # TODO: rewrite to parallelize
        # TODO: add cropped clip from dataset?
        # TODO: add rendering from CARLA
        # TODO: draw bones and not only dots?
        videos = []
        name_parts = []
        for clip_idx in range(rendered_videos):
            video = []
            openpose_clip = openpose[clip_idx]
            projection_clip = projection[clip_idx]
            video_id = video_ids[clip_idx]
            pedestrian_id = pedestrian_ids[clip_idx]
            clip_id = clip_ids[clip_idx]

            # easiest way to get (sparse) rendering is to re-calculate all pose changes
            pose_changes_clip = pose_change[clip_idx]
            bound_pedestrian: ControlledPedestrian = copy.deepcopy(
                self.__pedestrians[clip_idx])
            bound_pedestrian.bind(world)
            camera_queue = Queue()
            camera_rgb = setup_camera(
                world, camera_queue, bound_pedestrian)
            (prev_relative_loc, prev_relative_rot) = bound_pedestrian.current_pose.tensors
            # P3dPose.forward expects batches, so
            prev_relative_loc = prev_relative_loc.unsqueeze(0)
            prev_relative_rot = prev_relative_rot.unsqueeze(0)

            for fi, (openpose_frame, projection_frame, pose_change_frame) in enumerate(zip(openpose_clip, projection_clip, pose_changes_clip)):
                openpose_canvas = np.zeros((image_size[1], image_size[0], 4), np.uint8)
                openpose_img = self.__pose_projection.draw_projection_points(
                    openpose_canvas, openpose_frame, openpose_keys)

                projection_canvas = np.zeros(
                    (image_size[1], image_size[0], 4), np.uint8)
                projection_img = self.__pose_projection.draw_projection_points(
                    projection_canvas, projection_frame, projection_keys)

                # align them next to each other
                rgba_points_img = torch.tensor(np.concatenate(
                    (openpose_img, projection_img), axis=concatenation_axis))
                # blend alpha; TODO: alternatively overlap with clip/rendering?
                # points_img = ((rgba_points_img[..., 0:3] * torch.tensor([255], dtype=torch.float32)) /
                #               rgba_points_img[..., 3:4]).round().type(torch.uint8)
                # drop alpha
                points_img = rgba_points_img[..., 0:3]

                # get render frame from CARLA
                (_, _, prev_relative_rot) = bound_pedestrian.current_pose.forward(
                    pose_change_frame.detach().unsqueeze(0), prev_relative_loc, prev_relative_rot)

                bound_pedestrian.current_pose.tensors = (
                    prev_relative_loc[0], prev_relative_rot[0])
                bound_pedestrian.apply_pose()

                # TODO: teleport when implemented

                world_frame = world.tick()

                frames = []
                sensor_data = None

                carla_img = torch.zeros((*image_size, 3), dtype=torch.uint8)
                if world_frame:
                    # drain the sensor queue
                    try:
                        while (sensor_data is None) or sensor_data.frame < world_frame:
                            sensor_data = camera_queue.get(True, 1.0)
                            frames.append(sensor_data)
                    except Empty:
                        pass

                    if len(frames):
                        data = frames[-1]
                        data.convert(carla.ColorConverter.Raw)
                        img = PIL.Image.frombuffer('RGBA', (data.width, data.height),
                                                   data.raw_data, "raw", 'RGBA', 0, 1)  # load
                        img = img.convert('RGB')  # drop alpha
                        # the data is actually in BGR format, so switch channels
                        carla_img = torch.tensor(
                            np.array(img)[..., ::-1].copy(), dtype=torch.uint8)
                # end get render frame from CARLA

                # concatenate image from CARLA with others
                img = torch.cat((points_img, carla_img), dim=concatenation_axis)

                # TODO: add thin white border to separate groups from each other?

                video.append(img)  # H,W,C

            camera_rgb.stop()
            camera_rgb.destroy()

            # TODO: also destroy pedesstrian + increase spawn tries to be at least rendered_videos + 10

            video = torch.stack(video)  # T,H,W,C
            videos.append(video)
            name_parts.append((video_id, pedestrian_id, clip_id))

        # close connection to carla as needed - TODO: should this be in (logging) epoch end?
        if (client is not None) and (world is not None):
            destroy(client, world)
        return videos, name_parts
