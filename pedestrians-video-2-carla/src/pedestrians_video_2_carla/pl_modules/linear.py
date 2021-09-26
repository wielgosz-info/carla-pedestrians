import torch
from torch import nn
from torch.nn import functional as F

from pedestrians_video_2_carla.pl_modules.base import LitBaseMapper


class LitLinearMapper(LitBaseMapper):
    def __init__(self):
        super().__init__()

        self.pose_linear = nn.Linear(
            26 * 2,  # OpenPose mapped to CARLA points; what to do about NaNs?
            # bones rotations (euler angles; radians; roll, pitch, yaw) to get into the required position
            26 * 3
        )

    def forward(self, x):
        pose_change = self.pose_linear(x)
        return pose_change

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        pose_change = self.forward(train_batch)
        projected_pose = self._pose_projection(
            pose_change,
            torch.zeros((*train_batch.shape[:2], 3),
                        self.device),  # no world loc change
            torch.zeros((*train_batch.shape[:2], 3),
                        self.device),  # no world rot change
        )

        loss = F.mse_loss(projected_pose, train_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        pose_change = self.forward(val_batch)
        projected_pose = self._pose_projection(
            pose_change,
            torch.zeros((*val_batch.shape[:2], 3), self.device),  # no world loc change
            torch.zeros((*val_batch.shape[:2], 3), self.device),  # no world rot change
        )

        loss = F.mse_loss(projected_pose, val_batch)
        self.log('val_loss', loss)
