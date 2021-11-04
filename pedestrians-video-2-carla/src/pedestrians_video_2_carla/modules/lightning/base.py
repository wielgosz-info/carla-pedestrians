
from typing import Tuple

import pytorch_lightning as pl
from pedestrians_video_2_carla.modules.torch.projection import ProjectionModule
from pedestrians_video_2_carla.skeletons.nodes import MAPPINGS, Skeleton, get_skeleton_name_by_type, get_skeleton_type_by_name

from pedestrians_video_2_carla.skeletons.nodes.openpose import BODY_25_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from torch import nn
from torch.functional import Tensor
from enum import Enum


class LossModes(Enum):
    """
    Enum for loss modes.
    """
    common_loc_2d = 0
    loc_3d = 1
    loc_2d_3d = 2


class LitBaseMapper(pl.LightningModule):
    def __init__(
        self,
        input_nodes: Skeleton = BODY_25_SKELETON,
        output_nodes: Skeleton = CARLA_SKELETON,
        loss_mode: LossModes = LossModes.common_loc_2d,
        **kwargs
    ):
        super().__init__()

        self._loss_mode = loss_mode
        self._loss_fn = {
            LossModes.common_loc_2d: self._calculate_loss_common_loc_2d,
            LossModes.loc_3d: self._calculate_loss_loc_3d,
            LossModes.loc_2d_3d: self._calculate_loss_loc_2d_3d
        }[self._loss_mode]

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        # default layers
        self.projection = ProjectionModule(
            **kwargs
        )
        self.criterion = self._setup_criterion()

        self.save_hyperparameters({
            'input_nodes': get_skeleton_name_by_type(input_nodes),
            'output_nodes': get_skeleton_name_by_type(output_nodes),
            'loss_mode': self._loss_mode.name,
            'loss_criterion': '{}(reduction="{}")'.format(self.criterion.__class__.__name__, getattr(self.criterion, 'reduction', ''))
        })

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseMapper Lightning Module")
        parser.add_argument(
            '--input_nodes',
            type=get_skeleton_type_by_name,
            default=BODY_25_SKELETON
        )
        parser.add_argument(
            '--output_nodes',
            type=get_skeleton_type_by_name,
            default=CARLA_SKELETON
        )
        parser.add_argument(
            '--loss_mode',
            type=LossModes.__getitem__,
            metavar='{}'.format(set(LossModes.__members__.keys())),
            default=LossModes.common_loc_2d,
            choices=list(LossModes)
        )
        return parent_parser

    def _on_batch_start(self, batch, batch_idx, dataloader_idx):
        self.projection.on_batch_start(batch, batch_idx, dataloader_idx)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'test')

    def _step(self, batch, batch_idx, stage):
        (frames, targets, meta) = batch

        pose_change = self.forward(frames.to(self.device))

        (projected_pose, normalized_projection, absolute_pose_loc) = self.projection(
            pose_change
        )
        loss_dict = self._loss_fn(
            projected_pose=projected_pose,
            normalized_projection=normalized_projection,
            absolute_pose_loc=absolute_pose_loc,
            frames=frames,
            targets=targets,
            meta=meta
        )
        stage_loss_dict = {
            '{}_loss/{}'.format(stage, key): value for key, value in loss_dict.items()}

        for key, value in stage_loss_dict.items():
            self.log(key, value)

        self._log_videos(pose_change, projected_pose, batch, batch_idx, stage)

        return stage_loss_dict

    def _setup_criterion(self):
        return nn.MSELoss(reduction='mean')

    def _calculate_loss_common_loc_2d(self, normalized_projection, frames, **kwargs):
        if self.input_nodes == CARLA_SKELETON:
            carla_indices = slice(None)
            input_indices = slice(None)
        else:
            mappings = MAPPINGS[self.input_nodes]
            (carla_indices, input_indices) = zip(
                *[(c.value, o.value) for (c, o) in mappings])

        common_projection = normalized_projection[..., carla_indices, 0:2]
        common_input = frames[..., input_indices, 0:2]

        loss = self.criterion(
            common_projection,
            common_input
        )

        return {'common_loc_2d': loss}

    def _calculate_loss_loc_3d(self, absolute_pose_loc, targets, **kwargs):
        # TODO: coordinates are in meters, should we normalize? Especially when mixing different datasets
        loss = self.criterion(
            absolute_pose_loc,
            targets['absolute_pose_loc']
        )
        return {'loc_3d': loss}

    def _calculate_loss_loc_2d_3d(self, normalized_projection, absolute_pose_loc, frames, targets, **kwargs):
        loss = {}
        loss.update(self._calculate_loss_common_loc_2d(
            normalized_projection, frames, **kwargs))
        loss.update(self._calculate_loss_loc_3d(absolute_pose_loc, targets, **kwargs))

        loss.update(
            {'loc_2d_3d': loss['common_loc_2d'] + loss['loc_3d']}
        )

        return loss

    def _log_to_tensorboard(self, vid, vid_idx, fps, stage, meta):
        vid = vid.permute(0, 1, 4, 2, 3).unsqueeze(0)  # B,T,H,W,C -> B,T,C,H,W
        self.logger[0].experiment.add_video(
            '{}_{}_render'.format(stage, vid_idx),
            vid, self.global_step, fps=fps
        )

    def _log_videos(self, pose_change: Tensor, projected_pose: Tensor, batch: Tuple, batch_idx: int, stage: str, log_to_tb: bool = False):
        if log_to_tb:
            vid_callback = self._log_to_tensorboard
        else:
            vid_callback = None

        self.logger[1].experiment.log_videos(
            batch, projected_pose, pose_change, self.global_step, batch_idx, stage, vid_callback, force=(stage != 'train'))
