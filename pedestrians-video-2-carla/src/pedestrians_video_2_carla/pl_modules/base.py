
import os
from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision
from pedestrians_video_2_carla.pytorch_helpers.modules import ProjectionModule

from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO
from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON
from torch import nn
from torch.functional import Tensor


class LitBaseMapper(pl.LightningModule):
    def __init__(self, input_nodes: Union[BODY_25, COCO] = BODY_25, output_nodes=CARLA_SKELETON, **kwargs):
        super().__init__()

        # default layers
        self.projection = ProjectionModule(
            input_nodes,
            output_nodes,
            **kwargs
        )
        self.criterion = nn.MSELoss(reduction='mean')

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

        (common_openpose, common_projection, projected_pose, _) = self.projection(
            pose_change,
            frames
        )
        loss = self.criterion(
            common_projection,
            common_openpose
        )

        self.log('{}_loss'.format(stage), loss)
        self._log_videos(pose_change, projected_pose, batch, batch_idx, stage)

        return loss

    def _log_videos(self, pose_change: Tensor, projected_pose: Tensor, batch: Tuple, batch_idx: int, stage: str):
        if self.current_epoch % 20 > 0:
            # we only want to log every n-th epoch
            # TODO: make this configurable
            return

        # TODO: allow to specify how many videos from each batch should be created
        max_videos = 1
        videos_dir = os.path.join(self.logger.log_dir, 'videos', stage)
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)

        tb = self.logger.experiment
        fps = 30.0

        (videos, name_parts) = self.projection.render(batch,
                                                      projected_pose,
                                                      pose_change,
                                                      max_videos,
                                                      fps)

        # TODO: parallelize?
        for vid, name in zip(videos, name_parts):
            torchvision.io.write_video(
                os.path.join(videos_dir,
                             '{}-{}-{:0>2d}-ep{:0>4d}.mp4'.format(
                                 *name,
                                 self.current_epoch
                             )),
                vid,
                fps=fps
            )

        if stage != 'train':
            videos = torch.stack(videos).permute(
                0, 1, 4, 2, 3)  # B,T,H,W,C -> B,T,C,H,W
            tb.add_video('{}_{:0>2d}_gt_and_projection_points'.format(stage, batch_idx),
                         videos, self.global_step, fps=fps)
