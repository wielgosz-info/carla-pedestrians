import torch
from torch import nn

from pedestrians_video_2_carla.modules.lightning.base import LitBaseMapper


class LitLSTMMapper(LitBaseMapper):
    def __init__(self,
                 clip_length: int = 30,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.__clip_length = clip_length

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 3  # OpenPose (x,y,confidence) points

        self.__output_nodes_len = len(self.output_nodes)
        # bones rotations (euler angles; radians; roll, pitch, yaw) to get into the required position
        self.__output_features = 3

        self.__input_size = self.__input_nodes_len * self.__input_features
        self.__output_size = self.__output_nodes_len * self.__output_features

        hidden_size = 64
        num_layers = 2

        self.linear_1 = nn.Linear(
            self.__input_size,
            self.__input_size
        )
        self.lstm_1 = nn.LSTM(
            input_size=self.__input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear_2 = nn.Linear(hidden_size, self.__output_size)

    def forward(self, x):
        x = x.view(*x.shape[:-2], self.__input_size)
        x = self.linear_1(x)
        x, _ = self.lstm_1(x)
        pose_change = self.linear_2(x)
        pose_change = pose_change.reshape(
            (-1, self.__clip_length, self.__output_nodes_len, self.__output_features))
        return pose_change

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        return optimizer
