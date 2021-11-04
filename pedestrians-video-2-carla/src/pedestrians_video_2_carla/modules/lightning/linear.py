from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
import torch
from torch import nn

from pedestrians_video_2_carla.modules.lightning.base import LitBaseMapper


class Linear(LitBaseMapper):
    """
    The simplest dummy model used to debug the flow.
    """

    def __init__(self,
                 clip_length: int = 30,
                 **kwargs
                 ):
        super().__init__(needs_confidence=True, **kwargs)

        self.__clip_length = clip_length

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 3  # (x,y,confidence) points

        self.__output_nodes_len = len(self.output_nodes)
        # bones rotations
        self.__output_features = 6

        self.__input_size = self.__clip_length * self.__input_nodes_len * self.__input_features
        self.__output_size = self.__clip_length * self.__output_nodes_len * self.__output_features

        self.linear = nn.Linear(
            self.__input_size,
            self.__output_size
        )

    def forward(self, x):
        original_shape = x.shape
        x = x.view((-1, self.__input_size))
        pose_change = self.linear(x)
        pose_change = pose_change.view(*original_shape[0:2],
                                       self.__output_nodes_len, self.__output_features)
        return rotation_6d_to_matrix(pose_change, "XYZ")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
