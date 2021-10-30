from typing import Union

from torch.functional import Tensor
from pedestrians_video_2_carla.renderers.renderer import Renderer
from typing import List, Tuple
from torch.functional import Tensor
import numpy as np

from pedestrians_video_2_carla.skeletons.nodes.openpose import BODY_25_SKELETON, COCO_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.walker_control.pose_projection import PoseProjection


class PointsRenderer(Renderer):
    def __init__(self, input_nodes: Union[BODY_25_SKELETON, COCO_SKELETON, CARLA_SKELETON] = CARLA_SKELETON, **kwargs) -> None:
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
