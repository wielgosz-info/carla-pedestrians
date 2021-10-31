
import os
from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision
from pedestrians_video_2_carla.modules.torch.projection import ProjectionModule
from pedestrians_video_2_carla.skeletons.nodes import SKELETONS, get_skeleton_name_by_type, get_skeleton_type_by_name

from pedestrians_video_2_carla.skeletons.nodes.openpose import BODY_25_SKELETON, COCO_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from torch import nn
from torch.functional import Tensor
from pedestrians_video_2_carla.data import DATASETS_BASE, OUTPUTS_BASE


class LitBaseMapper(pl.LightningModule):
    def __init__(self, input_nodes: Union[BODY_25_SKELETON, COCO_SKELETON, CARLA_SKELETON] = BODY_25_SKELETON, output_nodes=CARLA_SKELETON, log_videos_every_n_epochs=10, enabled_renderers=None, **kwargs):
        super().__init__()

        # default layers
        self.projection = ProjectionModule(
            input_nodes,
            output_nodes,
            **kwargs
        )
        self.criterion = nn.MSELoss(reduction='mean')

        self.save_hyperparameters({
            'input_nodes': get_skeleton_name_by_type(input_nodes),
            'output_nodes': get_skeleton_name_by_type(output_nodes),
        })

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseMapper Lightning Module")
        parser.add_argument(
            '--input_nodes', type=get_skeleton_type_by_name, default='BODY_25_SKELETON')
        parser.add_argument(
            '--output_nodes', type=get_skeleton_type_by_name, default='CARLA_SKELETON')
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
        (frames, *_) = batch

        pose_change = self.forward(frames.to(self.device))

        (common_input, common_projection, projected_pose, _) = self.projection(
            pose_change,
            frames
        )
        loss = self.criterion(
            common_projection,
            common_input
        )

        self.log('{}_loss'.format(stage), loss)
        self._log_videos(pose_change, projected_pose, batch, batch_idx, stage)

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
