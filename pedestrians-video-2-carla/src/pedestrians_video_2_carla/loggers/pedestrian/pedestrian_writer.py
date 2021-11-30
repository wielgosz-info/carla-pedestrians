import os
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torchvision
from pedestrians_video_2_carla.renderers import MergingMethod
from pedestrians_video_2_carla.renderers.carla_renderer import CarlaRenderer
from pedestrians_video_2_carla.renderers.points_renderer import PointsRenderer
from pedestrians_video_2_carla.renderers.renderer import Renderer
from pedestrians_video_2_carla.renderers.source_videos_renderer import \
    SourceVideosRenderer
from pedestrians_video_2_carla.skeletons.nodes import Skeleton
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor
from pedestrians_video_2_carla.transforms.reference_skeletons import ReferenceSkeletonsDenormalize
from torch.functional import Tensor

from .pedestrian_renderers import PedestrianRenderers


class PedestrianWriter(object):
    def __init__(self,
                 log_dir: str,
                 renderers: List,
                 extractor: HipsNeckExtractor,
                 input_nodes: Skeleton,
                 output_nodes: Skeleton,
                 reduced_log_every_n_steps: int = 500,
                 fps: float = 30.0,
                 max_videos: int = 10,
                 merging_method: MergingMethod = MergingMethod.square,
                 source_videos_dir: str = None,
                 **kwargs) -> None:
        self._log_dir = log_dir

        self._reduced_log_every_n_steps = reduced_log_every_n_steps
        self._fps = fps
        self._max_videos = max_videos

        if self._max_videos > 0:
            self.__videos_slice = slice(0, self._max_videos)
        else:
            self.__videos_slice = slice(None)

        self._renderers = renderers
        if len(self._renderers) < 3 and merging_method == MergingMethod.square:
            merging_method = MergingMethod.horizontal
        self._merging_method = merging_method

        self._extractor = extractor

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes

        self.__denormalize = ReferenceSkeletonsDenormalize(
            extractor=self._extractor,
            autonormalize=False
        )

        # actual renderers
        self.__zeros_renderer = Renderer()

        self.__source_videos_renderer = SourceVideosRenderer(
            data_dir=source_videos_dir
        ) if PedestrianRenderers.source_videos in self._renderers else None

        self.__source_carla_renderer = CarlaRenderer(
            fps=self._fps
        ) if PedestrianRenderers.source_carla in self._renderers else None

        self.__input_renderer = PointsRenderer(
            input_nodes=self._input_nodes
        ) if PedestrianRenderers.input_points in self._renderers else None

        assert self._output_nodes == self._extractor.input_nodes, "Configuration mismatch, HipsNeckExtractor and PointsRenderer need to use the same Skeleton."
        self.__projection_renderer = PointsRenderer(
            input_nodes=self._output_nodes
        ) if PedestrianRenderers.projection_points in self._renderers else None

        self.__carla_renderer = CarlaRenderer(
            fps=self._fps
        ) if PedestrianRenderers.carla in self._renderers else None

    @torch.no_grad()
    def log_videos(self,
                   inputs: Tensor,
                   targets: Tensor,
                   meta: Tensor,
                   projected_pose: Tensor,
                   absolute_pose_loc: Tensor,
                   absolute_pose_rot: Tensor,
                   world_loc: Tensor,
                   world_rot: Tensor,
                   step: int,
                   batch_idx: int,
                   stage: str,
                   vid_callback: Callable = None,
                   force: bool = False,
                   **kwargs) -> None:
        if step % self._reduced_log_every_n_steps != 0 and not force:
            return

        for vid_idx, (vid, meta) in enumerate(self._render(
                inputs[self.__videos_slice],
                {k: v[self.__videos_slice] for k, v in targets.items()},
                {k: v[self.__videos_slice] for k, v in meta.items()},
                projected_pose[self.__videos_slice],
                absolute_pose_loc[self.__videos_slice],
                absolute_pose_rot[self.__videos_slice] if absolute_pose_rot is not None else None,
                world_loc[self.__videos_slice],
                world_rot[self.__videos_slice] if world_rot is not None else None,
                batch_idx)):
            video_dir = os.path.join(self._log_dir, stage, meta['video_id'])
            os.makedirs(video_dir, exist_ok=True)

            torchvision.io.write_video(
                os.path.join(video_dir,
                             '{pedestrian_id}-{clip_id:0>2d}-step={step:0>4d}.mp4'.format(
                                 **meta,
                                 step=step
                             )),
                vid,
                fps=self._fps
            )

            if vid_callback is not None:
                vid_callback(vid, vid_idx, self._fps, stage, meta)

    @torch.no_grad()
    def _render(self,
                frames: Tensor,
                targets: Tensor,
                meta: Dict[str, List[Any]],
                projected_pose: Tensor,
                absolute_pose_loc: Tensor,
                absolute_pose_rot: Tensor,
                world_loc: Tensor,
                world_rot: Tensor,
                batch_idx: int
                ) -> Iterator[Tuple[Tensor, Tuple[str, str, int]]]:
        """
        Prepares video data. **It doesn't save anything!**

        :param frames: Input frames
        :type frames: Tensor
        :param meta: Meta data for each clips
        :type meta: Dict[str, List[Any]]
        :param projected_pose: Output of the projection layer.
        :type projected_pose: Tensor
        :param absolute_pose_loc: Output from the .forward converted to absolute pose locations. Get it from projection layer.
        :type absolute_pose_loc: Tensor
        :param absolute_pose_rot: Output from the .forward converted to absolute pose rotations. May be None.
        :type absolute_pose_rot: Tensor
        :param world_loc: Output from the .forward converted to world locations. Get it from projection layer.
        :type world_loc: Tensor
        :param world_rot: Output from the .forward converted to world rotations. May be None.
        :type world_rot: Tensor
        :param batch_idx: Batch index
        :type batch_idx: int
        :return: List of videos and metadata
        :rtype: Tuple[List[Tensor], Tuple[str]]
        """

        # TODO: handle world_loc and world_rot in carla renderers
        # TODO: get this from LitBaseMapper projection layer
        # TODO: move those to the Renderers's constructors instead of .render
        image_size = (800, 600)
        fov = 90.0

        denormalized_frames = self.__denormalize.from_projection(frames, meta)

        source_videos = None
        # it only makes sense to render single source
        if self.__source_carla_renderer is not None and targets['absolute_pose_loc'] is not None:
            source_videos = self.__source_carla_renderer.render(
                targets['absolute_pose_loc'], targets['absolute_pose_rot'], targets['world_loc'], targets['world_rot'], meta, image_size
            )
        elif self.__source_videos_renderer is not None:
            source_videos = self.__source_videos_renderer.render(
                meta, image_size)

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
                absolute_pose_loc, absolute_pose_rot, world_loc, world_rot, meta, image_size
            )

        if self._merging_method == MergingMethod.square:
            source_videos = self.__zeros_renderer.render(
                frames, image_size) if source_videos is None else source_videos
            input_videos = self.__zeros_renderer.render(
                frames, image_size) if input_videos is None else input_videos
            projection_videos = self.__zeros_renderer.render(
                frames, image_size) if projection_videos is None else projection_videos
            carla_videos = self.__zeros_renderer.render(
                frames, image_size) if carla_videos is None else carla_videos
        else:
            rendered_videos = min(self._max_videos, len(frames))
            source_videos = [None] * \
                rendered_videos if source_videos is None else source_videos
            input_videos = [None] * \
                rendered_videos if input_videos is None else input_videos
            projection_videos = [
                None] * rendered_videos if projection_videos is None else projection_videos
            carla_videos = [None] * \
                rendered_videos if carla_videos is None else carla_videos

        for vid_idx, (input_vid, projection_vid, carla_vid, source_vid) in enumerate(zip(input_videos, projection_videos, carla_videos, source_videos)):
            if self._merging_method.value < 2:
                merged_vid = torch.tensor(np.concatenate(
                    [a for a in (source_vid, input_vid, projection_vid,
                                 carla_vid) if a is not None],
                    axis=self._merging_method.value+1
                ))
            else:  # square
                merged_vid = torch.tensor(
                    np.concatenate((
                        np.concatenate((source_vid, input_vid), axis=2),
                        np.concatenate((carla_vid, projection_vid), axis=2)
                    ), axis=1)
                )
            vid_meta = {
                'video_id': 'video_{:0>2d}_{:0>2d}'.format(
                    batch_idx,
                    vid_idx
                ),
                'pedestrian_id': '{}_{}'.format(meta['age'][vid_idx], meta['gender'][vid_idx]),
                'clip_id': 0
            }
            vid_meta.update({
                k: v[vid_idx]
                for k, v in meta.items()
            })
            yield merged_vid, vid_meta
