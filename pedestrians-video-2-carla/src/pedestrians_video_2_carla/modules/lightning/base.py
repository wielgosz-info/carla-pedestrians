
from typing import Dict, Tuple

import pytorch_lightning as pl
from pedestrians_video_2_carla.modules.torch.projection import ProjectionModule
from pedestrians_video_2_carla.skeletons.nodes import MAPPINGS, Skeleton, get_skeleton_name_by_type, get_skeleton_type_by_name

from pedestrians_video_2_carla.skeletons.nodes.openpose import BODY_25_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from torch import nn
from torch.functional import Tensor
from enum import Enum

from pedestrians_video_2_carla.transforms.hips_neck import CarlaHipsNeckExtractor, HipsNeckNormalize


class LossModes(Enum):
    """
    Enum for loss modes.
    """
    common_loc_2d = 0
    loc_3d = 1
    loc_2d_3d = 2
    pose_changes = 3


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
            LossModes.loc_2d_3d: self._calculate_loss_loc_2d_3d,
            LossModes.pose_changes: self._calculate_loss_pose_changes,
        }[self._loss_mode]

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        # default layers
        self.projection = ProjectionModule(
            **kwargs
        )
        self.criterions = self._setup_criterions()

        self.save_hyperparameters({
            'input_nodes': get_skeleton_name_by_type(input_nodes),
            'output_nodes': get_skeleton_name_by_type(output_nodes),
            'loss_mode': self._loss_mode.name,
            'loss_criterion': '{}(reduction="{}")'.format(
                self.criterions[self._loss_mode].__class__.__name__,
                getattr(self.criterions[self._loss_mode], 'reduction', '')
            )
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

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, None, 'train')

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._step(batch, batch_idx, dataloader_idx, 'test')

    def _step(self, batch, batch_idx, dataloader_idx, stage):
        (frames, targets, meta) = batch

        pose_changes = self.forward(frames.to(self.device))

        (projected_pose, normalized_projection, absolute_pose_loc, absolute_pose_rot) = self.projection(
            pose_changes
        )
        loss_dict = self._loss_fn(
            pose_changes=pose_changes,
            projected_pose=projected_pose,
            normalized_projection=normalized_projection,
            absolute_pose_loc=absolute_pose_loc,
            absolute_pose_rot=absolute_pose_rot,
            frames=frames,
            targets=targets,
            meta=meta
        )

        for k, v in loss_dict.items():
            self.log('{}_loss/{}'.format(stage, k.name), v)

        self._log_videos(pose_changes, projected_pose, batch,
                         batch_idx, dataloader_idx, stage)

        # return primary loss
        return loss_dict[self._loss_mode]

    def _setup_criterions(self):
        criterions = {
            k: nn.MSELoss(reduction='sum') if k == LossModes.pose_changes else nn.MSELoss(
                reduction='mean')
            for k in LossModes
        }
        return criterions

    def _calculate_loss_common_loc_2d(self, normalized_projection, frames, **kwargs) -> Dict[LossModes, Tensor]:
        carla_indices, input_indices = self._get_common_indices()

        common_projection = normalized_projection[..., carla_indices, 0:2]
        common_input = frames[..., input_indices, 0:2]

        loss = self.criterions[LossModes.common_loc_2d](
            common_projection,
            common_input
        )

        return {LossModes.common_loc_2d: loss}

    def _get_common_indices(self):
        if self.input_nodes == CARLA_SKELETON:
            carla_indices = slice(None)
            input_indices = slice(None)
        else:
            mappings = MAPPINGS[self.input_nodes]
            (carla_indices, input_indices) = zip(
                *[(c.value, o.value) for (c, o) in mappings])

        return carla_indices, input_indices

    def _calculate_loss_loc_3d(self, absolute_pose_loc, targets, **kwargs) -> Dict[LossModes, Tensor]:
        transform = HipsNeckNormalize(CarlaHipsNeckExtractor(self.input_nodes))
        loss = self.criterions[LossModes.loc_3d](
            transform(absolute_pose_loc, dim=3),
            transform(targets['absolute_pose_loc'], dim=3)
        )
        return {LossModes.loc_3d: loss}

    def _calculate_loss_loc_2d_3d(self, normalized_projection, absolute_pose_loc, frames, targets, **kwargs) -> Dict[LossModes, Tensor]:
        loss = {}
        loss.update(self._calculate_loss_common_loc_2d(
            normalized_projection, frames, **kwargs))
        loss.update(self._calculate_loss_loc_3d(absolute_pose_loc, targets, **kwargs))

        loss.update(
            {LossModes.loc_2d_3d: loss[LossModes.common_loc_2d] +
                loss[LossModes.loc_3d]}
        )

        return loss

    def _calculate_loss_pose_changes(self, pose_changes, targets, normalized_projection, absolute_pose_loc, frames, **kwargs) -> Dict[LossModes, Tensor]:
        loss = {
            LossModes.pose_changes: self.criterions[LossModes.pose_changes](
                pose_changes,
                targets['pose_changes']
            )
        }

        # TODO: add option to use loss like a metric (not used during the training) instead of this
        loss.update(self._calculate_loss_loc_2d_3d(
            normalized_projection, absolute_pose_loc, frames, targets))

        return loss

    def _log_to_tensorboard(self, vid, vid_idx, fps, stage, meta):
        vid = vid.permute(0, 1, 4, 2, 3).unsqueeze(0)  # B,T,H,W,C -> B,T,C,H,W
        self.logger[0].experiment.add_video(
            '{}_{}_render'.format(stage, vid_idx),
            vid, self.global_step, fps=fps
        )

    def _log_videos(self, pose_change: Tensor, projected_pose: Tensor, batch: Tuple, batch_idx: int, dataloader_idx: int, stage: str, log_to_tb: bool = False):
        if log_to_tb:
            vid_callback = self._log_to_tensorboard
        else:
            vid_callback = None

        self.logger[1].experiment.log_videos(
            batch, projected_pose, pose_change, self.global_step, batch_idx, dataloader_idx, stage, vid_callback, force=(stage != 'train'))
