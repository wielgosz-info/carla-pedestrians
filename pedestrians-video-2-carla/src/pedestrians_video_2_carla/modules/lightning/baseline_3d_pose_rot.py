import torch
from pedestrians_video_2_carla.modules.projection.projection import ProjectionTypes
from pedestrians_video_2_carla.submodules.baseline_3d_pose.model import \
    LinearModel as Baseline3DPoseModel
from torch import nn
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix

from .base import LitBaseMapper


class LitBaseline3DPoseRot(LitBaseMapper):
    """
    Based on the [PyTorch implementation](https://github.com/weigq/3d_pose_baseline_pytorch)
    of 3D pose baseline from the following paper:

    ```bibtex
    @inproceedings{martinez_2017_3dbaseline,
    title={A simple yet effective baseline for 3d human pose estimation},
    author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
    booktitle={ICCV},
    year={2017}
    }
    ```
    """

    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5,
                 **kwargs):
        super().__init__(
            projection_type=ProjectionTypes.absolute_loc_rot,
            **kwargs
        )

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 2  # (x, y) points

        self.__output_nodes_len = len(self.output_nodes)
        self.__output_features = 9  # (x, y, z) joints points + rotation 6d vector

        self.__input_size = self.__input_nodes_len * self.__input_features
        self.__output_size = self.__output_nodes_len * self.__output_features

        self.baseline = Baseline3DPoseModel(
            linear_size=linear_size,
            num_stage=2,
            p_dropout=0.5
        )

        # hack a little bit to get the model working with our data
        self.baseline.w1 = nn.Linear(self.__input_size, linear_size)
        self.baseline.w2 = nn.Linear(linear_size, self.__output_size)

        self.save_hyperparameters({
            'linear_size': linear_size,
            'num_stage': num_stage,
            'p_dropout': p_dropout
        })

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # the baseline model expects a single frame
        original_shape = x.shape
        x = x.view((-1, self.__input_size))
        x = self.baseline(x)
        x = x.view(*original_shape[0:2],
                   self.__output_nodes_len, self.__output_features)
        return x[..., :3], rotation_6d_to_matrix(x[..., 3:])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
