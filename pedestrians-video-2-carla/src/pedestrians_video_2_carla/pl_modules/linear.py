from typing import Union
import torch
from torch import nn
from torch.nn import functional as F

from pedestrians_video_2_carla.pl_modules.base import LitBaseMapper
from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO


class LitLinearMapper(LitBaseMapper):
    def __init__(self, points: Union[BODY_25, COCO] = BODY_25):
        super().__init__()

        self.pose_linear = nn.Linear(
            len(points) * 3,  # OpenPose (x,y,confidence) points
            # bones rotations (euler angles; radians; roll, pitch, yaw) to get into the required position
            26 * 3
        )

    def forward(self, x):
        pose_change = self.pose_linear(x)
        return pose_change

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.__step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.__step(batch, 'test')

    def __step(self, batch, stage):
        (_, _, frames) = batch
        pose_change = self.forward(frames)
        projected_pose = self._pose_projection(
            pose_change,
            torch.zeros((*frames.shape[:2], 3),
                        self.device),  # no world loc change
            torch.zeros((*frames.shape[:2], 3),
                        self.device),  # no world rot change
        )

        loss = F.mse_loss(projected_pose, frames)
        self.log('{}_loss'.format(stage), loss)
        return loss
