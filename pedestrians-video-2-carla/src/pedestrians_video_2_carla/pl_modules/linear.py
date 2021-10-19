from typing import Union
import torch
from torch import nn
from torch.nn import functional as F

from pedestrians_video_2_carla.pl_modules.base import LitBaseMapper
from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO, COMMON_NODES
from pedestrians_video_2_carla.pytorch_data.transforms import CarlaHipsNeckNormalize
from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON


class LitLinearMapper(LitBaseMapper):
    def __init__(self, points: Union[BODY_25, COCO] = BODY_25, clip_length: int = 30):
        super().__init__()

        self.__clip_length = clip_length
        self.__points = points

        self.__input_nodes = len(points)
        self.__input_features = 3  # OpenPose (x,y,confidence) points

        self.__output_nodes = 26  # TODO: pass this as arg?
        # bones rotations (euler angles; radians; roll, pitch, yaw) to get into the required position
        self.__output_features = 3

        self.__input_size = self.__clip_length * self.__input_nodes * self.__input_features
        self.__output_size = self.__clip_length * self.__output_nodes * self.__output_features

        self.pose_linear = nn.Linear(
            self.__input_size,
            self.__output_size
        )
        self.projection_transform = CarlaHipsNeckNormalize()

    def forward(self, x):
        x = x.reshape((-1, self.__input_size))
        pose_change = self.pose_linear(x)
        pose_change = pose_change.reshape(
            (-1, self.__clip_length, self.__output_nodes, self.__output_features))
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

        pose_change = self.forward(frames.to(self.device))
        projected_pose = self._pose_projection(
            pose_change,
            torch.zeros((*frames.shape[:2], 3),
                        device=self.device),  # no world loc change
            torch.zeros((*frames.shape[:2], 3),
                        device=self.device),  # no world rot change
        )
        normalized_projection = self.projection_transform(projected_pose)

        if type(self.__points) == COCO:
            mappings = COMMON_NODES['CARLA_2_COCO']
        else:  # default
            mappings = COMMON_NODES['CARLA_2_BODY_25']

        (carla_indices, openpose_indices) = zip(
            *[(c.value, o.value) for (c, o) in mappings])

        loss = F.mse_loss(
            normalized_projection[..., carla_indices, 0:2],
            frames[..., openpose_indices, 0:2]
        )
        self.log('{}_loss'.format(stage), loss)

        return loss
