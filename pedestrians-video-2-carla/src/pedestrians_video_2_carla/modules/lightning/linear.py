import torch
from torch import nn

from pedestrians_video_2_carla.modules.lightning.base import LitBaseMapper


class LitLinearMapper(LitBaseMapper):
    def __init__(self,
                 clip_length: int = 30,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.__clip_length = clip_length

        self.__input_nodes_len = len(self.projection.input_nodes)
        self.__input_features = 3  # OpenPose (x,y,confidence) points

        self.__output_nodes_len = len(self.projection.output_nodes)
        # bones rotations (euler angles; radians; roll, pitch, yaw) to get into the required position
        self.__output_features = 3

        self.__input_size = self.__clip_length * self.__input_nodes_len * self.__input_features
        self.__output_size = self.__clip_length * self.__output_nodes_len * self.__output_features

        self.pose_linear = nn.Linear(
            self.__input_size,
            self.__output_size
        )

    def forward(self, x):
        x = x.reshape((-1, self.__input_size))
        pose_change = self.pose_linear(x)
        pose_change = pose_change.reshape(
            (-1, self.__clip_length, self.__output_nodes_len, self.__output_features))
        return pose_change

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
