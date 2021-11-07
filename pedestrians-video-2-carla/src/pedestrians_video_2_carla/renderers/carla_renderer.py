import logging
from queue import Empty, Queue
from typing import Dict, List, Tuple

import carla
import numpy as np
import PIL
import torch
from pedestrians_video_2_carla.modules.projection.projection import ProjectionTypes
from pedestrians_video_2_carla.renderers.renderer import Renderer
from pedestrians_video_2_carla.carla_utils.destroy import destroy_client_and_world
from pedestrians_video_2_carla.carla_utils.setup import *
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from torch.functional import Tensor


class CarlaRenderer(Renderer):
    def __init__(self, fps=30.0, fov=90.0, projection_type=ProjectionTypes.pose_changes, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__fps = fps
        self.__fov = fov
        self.__projection_type = projection_type

    @torch.no_grad()
    def render(self, pose_change: Tensor, meta: List[Dict[str, Any]], image_size: Tuple[int, int] = (800, 600), **kwargs) -> List[np.ndarray]:
        rendered_videos = len(pose_change)

        # prepare connection to carla as needed - TODO: should this be in (logging) epoch start?
        client, world = setup_client_and_world(fps=self.__fps)

        for clip_idx in range(len(pose_change)):
            video = self.render_clip(
                pose_change[clip_idx],
                meta['age'][clip_idx],
                meta['gender'][clip_idx],
                image_size,
                world,
                rendered_videos
            )
            yield video

        # close connection to carla as needed - TODO: should this be in (logging) epoch end?
        if (client is not None) and (world is not None):
            destroy_client_and_world(client, world)

    @torch.no_grad()
    def render_clip(self, pose_changes_clip, age, gender, image_size, world, rendered_videos):
        # easiest way to get (sparse) rendering is to re-calculate all pose changes
        bound_pedestrian = ControlledPedestrian(
            world, age, gender, P3dPose, max_spawn_tries=10+rendered_videos, device=pose_changes_clip.device)
        camera_queue = Queue()
        camera_rgb = setup_camera(
            world, camera_queue, bound_pedestrian, image_size, self.__fov)
        (prev_relative_loc, prev_relative_rot) = bound_pedestrian.current_pose.tensors
        # P3dPose.forward expects batches, so
        prev_relative_loc = prev_relative_loc.unsqueeze(0)
        prev_relative_rot = prev_relative_rot.unsqueeze(0)

        video = []
        for pose_change_frame in pose_changes_clip:
            (frame, prev_relative_rot) = self.render_frame(pose_change_frame, prev_relative_loc, prev_relative_rot,
                                                           image_size, world, bound_pedestrian, camera_queue)
            video.append(frame)

        camera_rgb.stop()
        camera_rgb.destroy()

        bound_pedestrian.walker.destroy()

        return torch.stack(video, dim=0)

    @torch.no_grad()
    def render_frame(self,
                     pose_change_frame: Tensor,
                     prev_relative_loc: Tensor,
                     prev_relative_rot: Tensor,
                     image_size: Tuple[int, int],
                     world: carla.World,
                     bound_pedestrian: ControlledPedestrian,
                     camera_queue: Queue
                     ):
        if self.__projection_type == ProjectionTypes.pose_changes:
            (_, _, prev_relative_rot) = bound_pedestrian.current_pose.forward(
                pose_change_frame.detach().unsqueeze(0), prev_relative_loc, prev_relative_rot)

            bound_pedestrian.current_pose.tensors = (
                prev_relative_loc[0], prev_relative_rot[0])
            bound_pedestrian.apply_pose()
        elif self.__projection_type == ProjectionTypes.absolute_loc:
            prev_relative_rot = None
            abs_pose = bound_pedestrian.current_pose.empty
            for i, k in enumerate(abs_pose.keys()):
                abs_pose[k] = carla.Transform(
                    location=carla.Location(*pose_change_frame[i]))
            bound_pedestrian.apply_pose(abs_pose_snapshot=abs_pose)

        # TODO: teleport when implemented

        world_frame = world.tick()

        frames = []
        sensor_data = None

        carla_img = torch.zeros((image_size[1], image_size[0], 3), dtype=torch.uint8)
        if world_frame:
            # drain the sensor queue
            try:
                while (sensor_data is None) or sensor_data.frame < world_frame:
                    sensor_data = camera_queue.get(True, 1.0)
                    frames.append(sensor_data)
            except Empty:
                logging.getLogger(__name__).warn(
                    "Sensor data skipped in frame {}".format(world_frame))

            if len(frames):
                data = frames[-1]
                data.convert(carla.ColorConverter.Raw)
                img = PIL.Image.frombuffer('RGBA', (data.width, data.height),
                                           data.raw_data, "raw", 'RGBA', 0, 1)  # load
                img = img.convert('RGB')  # drop alpha
                # the data is actually in BGR format, so switch channels
                carla_img = torch.tensor(
                    np.array(img)[..., ::-1].copy(), dtype=torch.uint8)

        return (carla_img, prev_relative_rot)
