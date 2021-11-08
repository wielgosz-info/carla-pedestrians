import logging
from queue import Empty, Queue
from typing import Dict, List, Tuple, Union

import carla
import numpy as np
import PIL
import torch
from pedestrians_video_2_carla.renderers.renderer import Renderer
from pedestrians_video_2_carla.carla_utils.destroy import destroy_client_and_world
from pedestrians_video_2_carla.carla_utils.setup import *
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from torch.functional import Tensor
from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles


class CarlaRenderer(Renderer):
    def __init__(self, fps=30.0, fov=90.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__fps = fps
        self.__fov = fov

    @torch.no_grad()
    def render(self,
               absolute_pose_loc: Tensor,
               absolute_pose_rot: Tensor,
               meta: List[Dict[str, Any]],
               image_size: Tuple[int, int] = (800, 600),
               **kwargs
               ) -> List[np.ndarray]:
        rendered_videos = len(absolute_pose_loc)

        # prepare connection to carla as needed - TODO: should this be in (logging) epoch start?
        client, world = setup_client_and_world(fps=self.__fps)

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                absolute_pose_loc[clip_idx],
                absolute_pose_rot[clip_idx] if absolute_pose_rot is not None else None,
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
    def render_clip(self,
                    absolute_pose_loc_clip: Tensor,
                    absolute_pose_rot_clip: Union[Tensor, None],
                    age: str,
                    gender: str,
                    image_size: Tuple[int, int],
                    world: carla.World,
                    rendered_videos: int
                    ):
        bound_pedestrian = ControlledPedestrian(
            world, age, gender, P3dPose, max_spawn_tries=10+rendered_videos, device=absolute_pose_loc_clip.device)
        camera_queue = Queue()
        camera_rgb = setup_camera(
            world, camera_queue, bound_pedestrian, image_size, self.__fov)

        video = []
        for frame_idx, absolute_pose_loc_frame in enumerate(absolute_pose_loc_clip):
            absolute_pose_rot_frame = absolute_pose_rot_clip[
                frame_idx] if absolute_pose_rot_clip is not None else None
            frame = self.render_frame(absolute_pose_loc_frame, absolute_pose_rot_frame,
                                      image_size, world, bound_pedestrian, camera_queue)
            video.append(frame)

        camera_rgb.stop()
        camera_rgb.destroy()

        bound_pedestrian.walker.destroy()

        return torch.stack(video, dim=0)

    @torch.no_grad()
    def render_frame(self,
                     absolute_pose_loc_frame: Tensor,
                     absolute_pose_rot_frame: Tensor,
                     image_size: Tuple[int, int],
                     world: carla.World,
                     bound_pedestrian: ControlledPedestrian,
                     camera_queue: Queue
                     ):
        abs_pose = bound_pedestrian.current_pose.tensors_to_pose(
            absolute_pose_loc_frame, absolute_pose_rot_frame)
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

        return carla_img
