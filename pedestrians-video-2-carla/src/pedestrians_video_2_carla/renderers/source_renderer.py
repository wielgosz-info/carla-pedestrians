import glob
import logging
import math
import os
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pims
from pedestrians_video_2_carla.renderers.renderer import Renderer


class SourceRenderer(Renderer):
    def __init__(self, data_dir: str, set_filepath, **kwargs) -> None:
        super().__init__(**kwargs)

        self.__data_dir = data_dir
        self.__annotations = pd.read_csv(set_filepath)
        self.__annotations.set_index(['video', 'id'], inplace=True)
        self.__annotations.sort_index(inplace=True)

    def render(self, meta: List[Dict[str, Any]], stage: str = 'test', image_size: Tuple[int, int] = (800, 600), **kwargs) -> List[np.ndarray]:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        rendered_videos = min(self._max_videos, len(meta))

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                meta[clip_idx]['video_id'],
                meta[clip_idx]['pedestrian_id'],
                meta[clip_idx]['clip_id'],
                meta[clip_idx]['frame_id'][0],
                meta[clip_idx]['frame_ids'][1],
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
