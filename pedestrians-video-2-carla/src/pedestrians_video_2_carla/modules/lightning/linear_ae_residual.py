import torch
from torch import nn

from pedestrians_video_2_carla.modules.lightning.base import LitBaseMapper


class LitLinearAEResidual(LitBaseMapper):
    """
    Residual bottleneck autoencoder.
    Inputs are flattened to a vector of size (clip_length * input_nodes_len * input_features).
    """

    def __init__(self,
                 clip_length: int = 30,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.__clip_length = clip_length

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 2  # (x, y)

        self.__output_nodes_len = len(self.output_nodes)
        self.__output_features = 3

        self.__input_size = self.__clip_length * self.__input_nodes_len * self.__input_features
        self.__output_size = self.__clip_length * self.__output_nodes_len * self.__output_features

        self.__encoder = nn.Sequential(
            nn.Linear(self.__input_size, self.__input_size // 2),
            nn.BatchNorm1d(self.__input_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.__input_size // 2, self.__input_size // 4),
            nn.BatchNorm1d(self.__input_size // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.__input_size // 4, self.__input_size // 8),
            nn.BatchNorm1d(self.__input_size // 8),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.__decoder = nn.Sequential(
            nn.Linear(self.__input_size // 8, self.__output_size // 4),
            nn.BatchNorm1d(self.__output_size // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.__output_size // 4, self.__output_size // 2),
            nn.BatchNorm1d(self.__output_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.__output_size // 2, self.__output_size),
        )

        self.__residual_bottleneck = nn.Sequential(
           nn.Linear(self.__input_size, self.__input_size // 8),
           nn.BatchNorm1d(self.__input_size // 8),
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        original_shape = x.shape
        x = x.view((-1, self.__input_size))

        bottleneck = self.__encoder(x)
        x = bottleneck + self.__residual_bottleneck(x)
        pose_change = self.__decoder(x)

        pose_change = pose_change.view(*original_shape[0:2],
                                       self.__output_nodes_len, self.__output_features)
        return pose_change

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
