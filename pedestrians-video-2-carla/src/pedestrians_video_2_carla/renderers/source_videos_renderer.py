import glob
import logging
import math
import os
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import pims
from pedestrians_video_2_carla.renderers.renderer import Renderer


class SourceVideosRenderer(Renderer):
    def __init__(self, data_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)

        self.__data_dir = data_dir

    def render(self, meta: List[Dict[str, Any]], image_size: Tuple[int, int] = (800, 600), **kwargs) -> List[np.ndarray]:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        rendered_videos = len(meta['video_id'])

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                meta['video_id'][clip_idx],
                meta['pedestrian_id'][clip_idx],
                meta['clip_id'][clip_idx],
                meta['start_frame'][clip_idx],
                meta['end_frame'][clip_idx],
                meta['bboxes'][clip_idx],
                image_size
            )
            yield video

    def render_clip(self, video_id, pedestrian_id, clip_id, start_frame, end_frame, bboxes, image_size):
        (canvas_width, canvas_height) = image_size
        half_width = int(math.floor(canvas_width / 2))
        half_height = int(math.floor(canvas_height / 2))
        canvas = np.zeros((end_frame - start_frame, canvas_height,
                          canvas_width, 3), dtype=np.uint8)

        paths = glob.glob(os.path.join(self.__data_dir, '{}.*'.format(video_id)))
        try:
            assert len(paths) == 1
            video = pims.PyAVReaderTimed(paths[0])
            clip = video[start_frame:end_frame]

            centers = (bboxes.mean(dim=-2) + 0.5).round().cpu().numpy().astype(int)

            x_center = centers[..., 0]
            y_center = centers[..., 1]
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
