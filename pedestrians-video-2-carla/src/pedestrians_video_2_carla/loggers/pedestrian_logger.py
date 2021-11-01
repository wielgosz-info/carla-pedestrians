import logging
import os
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union

import numpy as np
import torch
import torchvision
from pedestrians_video_2_carla.data import DATASETS_BASE, OUTPUTS_BASE
from pedestrians_video_2_carla.renderers import MergingMethod
from pedestrians_video_2_carla.renderers.carla_renderer import CarlaRenderer
from pedestrians_video_2_carla.renderers.points_renderer import PointsRenderer
from pedestrians_video_2_carla.renderers.renderer import Renderer
from pedestrians_video_2_carla.renderers.source_videos_renderer import \
    SourceVideosRenderer
from pedestrians_video_2_carla.skeletons.nodes import Skeleton
from pedestrians_video_2_carla.transforms.hips_neck import (
    CarlaHipsNeckExtractor, HipsNeckDeNormalize, HipsNeckExtractor)
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from pedestrians_video_2_carla.walker_control.torch.pose_projection import \
    P3dPoseProjection
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from torch.functional import Tensor


class PedestrianRenderers(Enum):
    none = 0
    source_videos = 1
    source_carla = 2
    input_points = 3
    projection_points = 4
    carla = 5


class DisabledPedestrianWriter(object):
    def __init__(self, *args, **kwargs):
        pass

    def log_videos(self, *args, **kwargs):
        pass


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

        self.__reference_projections = None

        # actual renderers
        self.__zeros_renderer = Renderer()

        self.__source_videos_renderer = SourceVideosRenderer(
            # TODO: make this a parameter
            data_dir=os.path.join(DATASETS_BASE, 'JAAD', 'videos'),
            # TODO: make this a parameter
            set_filepath=os.path.join(OUTPUTS_BASE, 'JAAD', 'videos')
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

    def log_videos(self,
                   batch: Tensor,
                   projected_pose: Tensor,
                   pose_change: Tensor,
                   step: int,
                   batch_idx: int,
                   stage: str,
                   vid_callback: Callable = None,
                   force: bool = False,
                   **kwargs) -> None:
        if step % self._reduced_log_every_n_steps != 0 and not force:
            return

        for vid_idx, (vid, meta) in enumerate(self._render(
                batch[0][self.__videos_slice],
                batch[1]['pose_changes'][self.__videos_slice] if batch[1] is not None else None,
                {k: v[self.__videos_slice] for k, v in batch[2].items()},
                projected_pose[self.__videos_slice],
                pose_change[self.__videos_slice],
                batch_idx,
                stage)):
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

    def _denormalize(self, frames: Tensor, meta: Dict[str, List[Any]]) -> Tensor:
        # shortcut - we have only 4 possible skeletons
        if self.__reference_projections is None:
            types, projections = self._get_reference_projections(device=frames.device)
            self.__reference_projections = {
                t: p for t, p in zip(types, projections)
            }

        frame_projections = torch.stack([
            self.__reference_projections[(age, gender)]
            for (age, gender) in zip(meta['age'], meta['gender'])
        ], dim=0)

        return HipsNeckDeNormalize().from_projection(self._extractor, frame_projections)(frames)

    def _get_reference_projections(self, device):
        types = [
            ('adult', 'female'),
            ('adult', 'male'),
            ('child', 'female'),
            ('child', 'male')
        ]
        pedestrians = [
            ControlledPedestrian(age=age, gender=gender,
                                 pose_cls=P3dPose, device=device)
            for (age, gender) in types
        ]

        movements = torch.zeros(
            (len(pedestrians), 1, len(self._output_nodes), 3), device=device)
        (relative_loc, relative_rot) = zip(*[
            p.current_pose.tensors
            for p in pedestrians
        ])

        # TODO: get camera settings from LitBaseMapper.projection
        pose_projection = P3dPoseProjection(device=device, pedestrian=pedestrians[0])

        # TODO: we're assuming no in-world movement for now!
        world_locations = torch.zeros(
            (len(pedestrians), 3), device=device)
        world_rotations = torch.eye(3, device=device).reshape(
            (1, 3, 3)).repeat((len(pedestrians), 1, 1))

        return types, pose_projection.forward(
            pedestrians[0].current_pose.forward(
                movements,
                torch.stack(relative_loc),
                torch.stack(relative_rot)
            )[0],
            world_locations,
            world_rotations
        ).unsqueeze(1)

    def _render(self,
                frames: Tensor,
                targets: Tensor,
                meta: Dict[str, List[Any]],
                projected_pose: Tensor,
                pose_change: Tensor,
                batch_idx: int,
                stage: str
                ) -> Iterator[Tuple[Tensor, Tuple[str, str, int]]]:
        """
        Prepares video data. **It doesn't save anything!**

        :param frames: Input frames
        :type frames: Tensor
        :param meta: Meta data for each clips
        :type meta: Dict[str, List[Any]]
        :param projected_pose: Output of the projection layer
        :type projected_pose: Tensor
        :param pose_change: Output from the .forward
        :type pose_change: Tensor
        :param batch_idx: Batch index
        :type batch_idx: int
        :param stage: train/val/test/predict
        :type stage: str
        :return: List of videos and metadata
        :rtype: Tuple[List[Tensor], Tuple[str]]
        """

        # TODO: get this from LitBaseMapper projection layer
        # TODO: move those to the Renderers's constructors instead of .render
        image_size = (800, 600)
        fov = 90.0

        denormalized_frames = self._denormalize(frames, meta)

        source_videos = None
        # it only makes sense to render single source
        if self.__source_carla_renderer is not None and targets is not None:
            source_videos = self.__source_carla_renderer.render(
                targets, meta, image_size
            )
        elif self.__source_videos_renderer is not None:
            source_videos = self.__source_videos_renderer.render(
                meta, stage, image_size)

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
                pose_change, meta, image_size
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
                'video_id': 'video_{:0>2d}_{:0>2d}'.format(batch_idx, vid_idx),
                'pedestrian_id': '{}_{}'.format(meta['age'][vid_idx], meta['gender'][vid_idx]),
                'clip_id': 0
            }
            vid_meta.update({k: v[vid_idx] for k, v in meta.items()})
            yield merged_vid, vid_meta


class PedestrianLogger(LightningLoggerBase):
    """
    Logger for video output.
    """

    def __init__(self,
                 save_dir: str,
                 name: str,
                 version: Union[int, str] = 0,
                 log_every_n_steps: int = 50,
                 video_saving_frequency_reduction: int = 10,
                 renderers: List[PedestrianRenderers] = None,
                 extractor: HipsNeckExtractor = None,
                 **kwargs):
        """
        Initialize PedestrianLogger.

        :param save_dir: Directory to save videos. Usually you get this from TensorBoardLogger.log_dir + 'videos'.
        :type save_dir: str
        :param name: Name of the experiment.
        :type name: str
        :param version: Version of the experiment.
        :type version: Union[int, str]
        :param log_every_n_steps: Log every n steps. This should match the Trainer setting,
            the final value will be determined by multiplying it with video_saving_frequency_reduction. Default: 50.
        :type log_every_n_steps: int
        :param video_saving_frequency_reduction: Reduce the video saving frequency by this factor. Default: 10.
        :type video_saving_frequency_reduction: int
        :param renderers: List of used renderers. Default: ['input_points', 'projection_points'].
        :type renderers: List[PedestrianRenderers]
        :param extractor: Extractor used for denormalization. Default: CarlaHipsNeckExtractor().
        :type extractor: HipsNeckExtractor
        """
        super().__init__(
            agg_key_funcs=kwargs.get('agg_key_funcs', None),
            agg_default_func=kwargs.get('agg_default_func', np.mean),
        )

        self._save_dir = save_dir
        self._name = name
        self._version = version
        self._kwargs = kwargs
        self._experiment = None
        self._writer_cls = PedestrianWriter

        self._video_saving_frequency_reduction = video_saving_frequency_reduction
        self._log_every_n_steps = log_every_n_steps
        self._reduced_log_every_n_steps = self._log_every_n_steps * \
            self._video_saving_frequency_reduction

        if self._reduced_log_every_n_steps <= 1:
            logging.getLogger(__name__).warning(
                "Video logging interval set to 0. Disabling video output.")
            self._writer_cls = DisabledPedestrianWriter

        # If renderers were not specified, use default. To disable, 'none' renderer must be passed explicitly.
        self._renderers = list(set(renderers)) if (renderers is not None) and (len(renderers) > 0) else [
            PedestrianRenderers.input_points, PedestrianRenderers.projection_points
        ]

        try:
            self._renderers.remove(PedestrianRenderers.none)
        except ValueError:
            pass

        if len(self._renderers) == 0:
            logging.getLogger(__name__).warning(
                "No renderers specified. Disabling video output.")
            self._writer_cls = DisabledPedestrianWriter

        if extractor is None:
            extractor = CarlaHipsNeckExtractor()
        self._extractor = extractor

    @staticmethod
    def add_logger_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Pedestrian Logger")
        parser.add_argument(
            "--video_saving_frequency_reduction",
            dest="video_saving_frequency_reduction",
            help="Set video saving frequency reduction to save them every REDUCTION * log_every_n_steps. If 0 or less disables saving videos.",
            metavar="REDUCTION",
            default=10,
            type=lambda x: max(int(x), 0)
        )
        parser.add_argument(
            "--max_videos",
            dest="max_videos",
            help="Set maximum number of videos to save from each batch. Set to -1 to save all videos in batch. Default: 10",
            default=10,
            type=int
        )
        parser.add_argument(
            "--renderers",
            dest="renderers",
            help="""
                Set renderers to use for video output.
                To disable rendering, 'none' renderer must be explicitly passed as the only renderer.
                Choices: {}.
                Default: ['input_points', 'projection_points']
                """.format(
                set(PedestrianRenderers.__members__.keys())),
            metavar="RENDERER",
            default=[],
            choices=list(PedestrianRenderers),
            nargs="+",
            action="extend",
            type=PedestrianRenderers.__getitem__
        )

        return parent_parser

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        return self._version

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            self._experiment = self._writer_cls(
                log_dir=self._save_dir,
                renderers=self._renderers,
                reduced_log_every_n_steps=self._reduced_log_every_n_steps,
                extractor=self._extractor,
                **self._kwargs
            )

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass
