import logging
from queue import Empty, Queue
from typing import Dict, List, Tuple, Union

import warnings
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
from pytorch_lightning.utilities import rank_zero_warn
from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles


try:
    import carla
except ImportError:
    import pedestrians_video_2_carla.carla_utils.mock_carla as carla
    warnings.warn("Using mock carla.", ImportWarning)


class CarlaRenderer(Renderer):
    def __init__(self, fps=30.0, fov=90.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__fps = fps
        self.__fov = fov

    @torch.no_grad()
    def render(self,
               absolute_pose_loc: Tensor,
               absolute_pose_rot: Tensor,
               world_loc: Tensor,
               world_rot: Tensor,
               meta: List[Dict[str, Any]],
               image_size: Tuple[int, int] = (800, 600),
               **kwargs
               ) -> List[np.ndarray]:
        rendered_videos = len(absolute_pose_loc)

        if absolute_pose_rot is None:
            rank_zero_warn(
                "Absolute pose rotations are not available, falling back to reference rotations. " +
                "Please note that this may result in weird rendering effects.")

        # prepare connection to carla as needed - TODO: should this be in (logging) epoch start?
        client, world = setup_client_and_world(fps=self.__fps)

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                absolute_pose_loc[clip_idx],
                absolute_pose_rot[clip_idx] if absolute_pose_rot is not None else None,
                world_loc[clip_idx],
                world_rot[clip_idx],
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
                    world_loc_clip: Tensor,
                    world_rot_clip: Tensor,
                    age: str,
                    gender: str,
                    image_size: Tuple[int, int],
                    world: 'carla.World',
                    rendered_videos: int
                    ):
        bound_pedestrian = ControlledPedestrian(
            world, age, gender, P3dPose, max_spawn_tries=10+rendered_videos, device=absolute_pose_loc_clip.device)
        camera_queue = Queue()
        camera_rgb = setup_camera(
            world, camera_queue, bound_pedestrian, image_size, self.__fov)

        if absolute_pose_rot_clip is None:
            # for correct rendering, the rotations are required
            # since we don't have them, try to salvage the situation
            # by using the rotations from reference pose
            (_, ref_abs_pose_rot) = bound_pedestrian.current_pose.pose_to_tensors(
                bound_pedestrian.current_pose.absolute)
            absolute_pose_rot_clip = [
                ref_abs_pose_rot
            ] * len(absolute_pose_loc_clip)

        video = []
        for absolute_pose_loc_frame, absolute_pose_rot_frame, world_loc_frame, world_rot_frame in zip(absolute_pose_loc_clip, absolute_pose_rot_clip, world_loc_clip, world_rot_clip):
            frame = self.render_frame(absolute_pose_loc_frame, absolute_pose_rot_frame,
                                      world_loc_frame, world_rot_frame,
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
                     world_loc_frame: Tensor,
                     world_rot_frame: Tensor,
                     image_size: Tuple[int, int],
                     world: 'carla.World',
                     bound_pedestrian: ControlledPedestrian,
                     camera_queue: Queue
                     ):
        abs_pose = bound_pedestrian.current_pose.tensors_to_pose(
            absolute_pose_loc_frame, absolute_pose_rot_frame)
        bound_pedestrian.apply_pose(abs_pose_snapshot=abs_pose)

        world_loc = world_loc_frame.cpu().numpy().astype(float)
        world_rot = -np.rad2deg(matrix_to_euler_angles(world_rot_frame,
                                "XYZ").cpu().numpy()).astype(float)

        bound_pedestrian.teleport_by(
            carla.Transform(
                carla.Location(
                    x=world_loc[0],
                    y=world_loc[1],
                    z=-world_loc[2]
                ),
                carla.Rotation(
                    pitch=world_rot[1],
                    yaw=world_rot[2],
                    roll=world_rot[0]
                )
            ),
            from_initial=True
        )

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
