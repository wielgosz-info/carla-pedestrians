import torch
from torch import nn

from pedestrians_video_2_carla.modules.lightning.base import LitBaseMapper


class LitLinearAutoencoderMapper(LitBaseMapper):
    def __init__(self,
                 clip_length: int = 30,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.__clip_length = clip_length

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 3

        self.__output_nodes_len = len(self.output_nodes)
        self.__output_features = 3

        self.__input_size = self.__clip_length * self.__input_nodes_len * self.__input_features
        self.__output_size = self.__clip_length * self.__output_nodes_len * self.__output_features

        self.__encoder = nn.Sequential(
            nn.Linear(self.__input_size, self.__input_size // 2),
            nn.ReLU(),
            nn.Linear(self.__input_size // 2, self.__input_size // 4),
            nn.ReLU(),
            nn.Linear(self.__input_size // 4, self.__input_size // 8),
            nn.ReLU(),
        )

        self.__decoder = nn.Sequential(
            nn.Linear(self.__input_size // 8, self.__output_size // 4),
            nn.ReLU(),
            nn.Linear(self.__output_size // 4, self.__output_size // 2),
            nn.ReLU(),
            nn.Linear(self.__output_size // 2, self.__output_size),
        )

    def forward(self, inputs):
        x = inputs.reshape((-1, self.__input_size))

        x = self.__encoder(x)
        x = self.__decoder(x)

        pose_change = x.reshape(
            (-1, self.__clip_length, self.__output_nodes_len, self.__output_features))
        return pose_change

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
