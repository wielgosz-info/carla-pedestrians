import glob
import math
import os
from enum import Enum
from queue import Empty, Queue
from typing import List, Tuple, Union
import logging
import warnings

import carla
import numpy as np
import pandas
import PIL
import pims
import torch
from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.utils.destroy import destroy
from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO
from pedestrians_video_2_carla.utils.setup import (setup_camera,
                                                   setup_client_and_world)
from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.pose_projection import \
    PoseProjection
from torch.functional import Tensor


class AlphaBehavior(Enum):
    drop = 0
    blend = 1
    keep = 2


class MergingMethod(Enum):
    vertical = 0
    horizontal = 1
    square = 2


class Renderer(object):
    def __init__(self, max_videos: int = 1, alpha: AlphaBehavior = AlphaBehavior.drop, **kwargs) -> None:
        self._max_videos = max_videos
        self._alpha = alpha

    def render(self, frames: Tensor, image_size: Tuple[int, int] = (800, 600), **kwargs) -> List[np.ndarray]:
        """
        Renders black clip and repeats it N times in the output.
        Number of channels depends on the defined alpha behavior.

        :param frames: Batch of sequence data (N, L, ...)
        :type frames: Tensor
        :return: List with size N of black clips (L, height, width, channels)
        :rtype: List[np.ndarray]
        """
        rendered_videos = min(self._max_videos, len(frames))
        for _ in range(rendered_videos):
            yield self.alpha_behavior(np.zeros((frames.shape[1], image_size[1], image_size[0], 4)))

    def alpha_behavior(self, rgba_frames: np.ndarray) -> np.ndarray:
        if self._alpha == AlphaBehavior.drop:
            return rgba_frames[..., 0:3]
        elif self._alpha == AlphaBehavior.blend:
            return ((rgba_frames[..., 0:3] * 255.0) / rgba_frames[..., 3:4]).round().type(np.uint8)
        # keep
        return rgba_frames


class PointsRenderer(Renderer):
    def __init__(self, input_nodes: Union[BODY_25, COCO, CARLA_SKELETON] = CARLA_SKELETON, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__keys = [k.name for k in input_nodes]

    def render(self, frames: Tensor, image_size: Tuple[int, int] = (800, 600), **kwargs) -> List[np.ndarray]:
        rendered_videos = min(self._max_videos, len(frames))
        cpu_frames = frames[..., 0:2].round().int().cpu().numpy()

        for clip_idx in range(rendered_videos):
            video = self.render_clip(cpu_frames[clip_idx], image_size)
            yield video

    def render_clip(self, openpose_clip: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        video = []

        for openpose_frame in openpose_clip:
            frame = self.render_frame(openpose_frame, image_size)
            video.append(frame)

        return self.alpha_behavior(np.stack(video))

    def render_frame(self, openpose_frame: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        canvas = np.zeros(
            (image_size[1], image_size[0], 4), np.uint8)
        # TODO: draw bones and not only dots?
        rgba_frame = PoseProjection.draw_projection_points(
            canvas, openpose_frame, self.__keys)

        return rgba_frame


class CarlaRenderer(Renderer):
    def __init__(self, fps=30.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__fps = 30.0

    def render(self, pose_change: Tensor, ages: List[str], genders: List[str], image_size: Tuple[int, int] = (800, 600), **kwargs) -> List[np.ndarray]:
        rendered_videos = min(self._max_videos, len(pose_change))

        # prepare connection to carla as needed - TODO: should this be in (logging) epoch start?
        client, world = setup_client_and_world(fps=self.__fps)

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                pose_change[clip_idx],
                ages[clip_idx],
                genders[clip_idx],
                image_size,
                world,
                rendered_videos
            )
            yield video

        # close connection to carla as needed - TODO: should this be in (logging) epoch end?
        if (client is not None) and (world is not None):
            destroy(client, world)

    def render_clip(self, pose_changes_clip, age, gender, image_size, world, rendered_videos):
        # easiest way to get (sparse) rendering is to re-calculate all pose changes
        bound_pedestrian = ControlledPedestrian(
            world, age, gender, P3dPose, max_spawn_tries=10+rendered_videos)
        camera_queue = Queue()
        camera_rgb = setup_camera(
            world, camera_queue, bound_pedestrian)
        (prev_relative_loc, prev_relative_rot) = bound_pedestrian.current_pose.tensors
        # P3dPose.forward expects batches, so
        prev_relative_loc = prev_relative_loc.unsqueeze(0)
        prev_relative_rot = prev_relative_rot.unsqueeze(0)

        video = []
        for pose_change_frame in pose_changes_clip:
            frame = self.render_frame(pose_change_frame, prev_relative_loc, prev_relative_rot,
                                      image_size, world, bound_pedestrian, camera_queue)
            video.append(frame)

        camera_rgb.stop()
        camera_rgb.destroy()

        bound_pedestrian.walker.destroy()

        return video

    def render_frame(self,
                     pose_change_frame: Tensor,
                     prev_relative_loc: Tensor,
                     prev_relative_rot: Tensor,
                     image_size: Tuple[int, int],
                     world: carla.World,
                     bound_pedestrian: ControlledPedestrian,
                     camera_queue: Queue
                     ):
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


class SourceRenderer(Renderer):
    def __init__(self, data_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)

        self.__data_dir = data_dir
        # TODO: figure somewhat better how to get this or (better) use the dumped dataset files
        self.__annotations = pandas.read_csv(
            os.path.join('/outputs', 'JAAD', 'annotations.csv'))
        self.__annotations.set_index(['video', 'id'], inplace=True)
        self.__annotations.sort_index(inplace=True)

    def render(self, video_ids: List[str], pedestrian_ids: List[str], clip_ids: List[int], frame_ids: Tuple[List[int], List[int]], stage: str = 'test', image_size: Tuple[int, int] = (800, 600), **kwargs) -> List[np.ndarray]:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        rendered_videos = min(self._max_videos, len(video_ids))

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                video_ids[clip_idx],
                pedestrian_ids[clip_idx],
                int(clip_ids[clip_idx]),
                int(frame_ids[0][clip_idx]),
                int(frame_ids[1][clip_idx]),
                image_size
            )
            yield video

    def render_clip(self, video_id, pedestrian_id, clip_id, frame_start, frame_stop, image_size):
        (canvas_width, canvas_height) = image_size
        half_width = int(math.floor(canvas_width / 2))
        half_height = int(math.floor(canvas_height / 2))
        canvas = np.zeros((frame_stop - frame_start, canvas_height,
                          canvas_width, 3), dtype=np.uint8)

        paths = glob.glob(os.path.join(self.__data_dir, '{}.*'.format(video_id)))
        try:
            assert len(paths) == 1
            video = pims.PyAVReaderTimed(paths[0])
            clip = video[frame_start:frame_stop]

            bboxes = self.__annotations.loc[
                (video_id, pedestrian_id)
            ][['frame', 'x1', 'x2', 'y1', 'y2']]

            bboxes = bboxes.loc[
                (bboxes['frame'] >= frame_start) &
                (bboxes['frame'] < frame_stop)
            ].set_index(['frame']).sort_index()

            x_center = (bboxes[['x1', 'x2']].to_numpy().mean(
                axis=1) + 0.5).round(0).astype(int)
            y_center = (bboxes[['y1', 'y2']].to_numpy().mean(
                axis=1) + 0.5).round(0).astype(int)
            (clip_height, clip_width, _) = clip.frame_shape

            # the most primitive solution for now - just cut off a piece around center point
            for idx in range(len(clip)):
                frame_x_min = int(max(0, x_center[idx]-half_width))
                frame_x_max = int(min(clip_width, x_center[idx]+half_width))
                frame_y_min = int(max(0, y_center[idx]-half_height))
                frame_y_max = int(min(clip_height, y_center[idx]+half_height))
                frame_width = frame_x_max - frame_x_min
                frame_height = frame_y_max - frame_y_min
                canvas_x_shift = max(0, half_width-x_center[idx])
                canvas_y_shift = max(0, half_height-y_center[idx])
                canvas[idx, canvas_y_shift:canvas_y_shift+frame_height, canvas_x_shift:canvas_x_shift +
                       frame_width] = clip[idx][frame_y_min:frame_y_max, frame_x_min:frame_x_max]

        except AssertionError:
            # no video or multiple candidates - skip
            logging.getLogger(__name__).warn(
                "Clip extraction failed for {}, {}, {}".format(video_id, pedestrian_id, clip_id))

        return canvas
