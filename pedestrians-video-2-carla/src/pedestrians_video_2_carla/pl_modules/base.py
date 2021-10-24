
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

        self.__fps = 30.0

        # default layers
        self.projection = ProjectionModule(
            input_nodes,
            output_nodes,
            fps=self.__fps,
            max_videos=64,
            enabled_renderers={
                'source': True,
                'input': True,
                'projection': True,
                'carla': False
            },
            data_dir=os.path.join('/datasets', 'JAAD'),  # TODO: get it from datamodule
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

    def _log_videos(self, pose_change: Tensor, projected_pose: Tensor, batch: Tuple, batch_idx: int, stage: str, log_to_tb: bool = False):
        if stage == 'train':
            # never log videos during training
            return

        if self.global_step % self.trainer.log_every_n_steps != 0:
            return

        videos_dir = os.path.join(self.logger.log_dir, 'videos', stage)
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)

        for vid_idx, (vid, name) in enumerate(self.projection.render(batch,
                                                                     projected_pose,
                                                                     pose_change,
                                                                     stage)):
            torchvision.io.write_video(
                os.path.join(videos_dir,
                             '{}-{}-{:0>2d}-ep{:0>4d}.mp4'.format(
                                 *name,
                                 self.current_epoch
                             )),
                vid,
                fps=self.__fps
            )

            if log_to_tb:
                tb = self.logger.experiment
                vid = vid.permute(
                    0, 1, 4, 2, 3).unsqueeze(0)  # B,T,H,W,C -> B,T,C,H,W
                tb.add_video('{}_{:0>2d}_{}_render'.format(stage, batch_idx, vid_idx),
                             vid, self.global_step, fps=self.__fps)
