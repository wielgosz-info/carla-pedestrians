import torch
from pedestrians_video_2_carla.modules.base.movements import MovementsModel
from pedestrians_video_2_carla.modules.base.output_types import MovementsModelOutputType
from pedestrians_video_2_carla.submodules.baseline_3d_pose.model import \
    LinearModel as Baseline3DPoseModel
from torch import nn
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Baseline3DPoseRot(MovementsModel):
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

    Main change is the addition of the rotation part in the output.
    """

    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5,
                 **kwargs):
        super().__init__(**kwargs)

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 2  # (x, y) points

        self.__output_nodes_len = len(self.output_nodes)
        self.__output_features = 9  # (x, y, z) joints points + rotation 6d vector

        self.__input_size = self.__input_nodes_len * self.__input_features
        self.__output_size = self.__output_nodes_len * self.__output_features

        self.baseline = Baseline3DPoseModel(
            linear_size=linear_size,
            num_stage=num_stage,
            p_dropout=p_dropout
        )

        # hack a little bit to get the model working with our data
        self.baseline.w1 = nn.Linear(self.__input_size, linear_size)
        self.baseline.w2 = nn.Linear(linear_size, self.__output_size)

        self._hparams = {
            'linear_size': linear_size,
            'num_stage': num_stage,
            'p_dropout': p_dropout
        }

        self.apply(self.init_weights)

    @property
    def output_type(self) -> MovementsModelOutputType:
        return MovementsModelOutputType.absolute_loc_rot

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Baseline3DPoseRot Movements Module")
        parser.add_argument(
            '--num_stage',
            default=2,
            type=int,
        )
        parser.add_argument(
            '--linear_size',
            default=1024,
            type=int,
        )
        parser.add_argument(
            '--p_dropout',
            default=0.5,
            type=float,
        )
        return parent_parser

    def forward(self, x, *args, **kwargs):
        # the baseline model expects a single frame
        original_shape = x.shape
        x = x.view((-1, self.__input_size))
        x = self.baseline(x)
        x = x.view(*original_shape[0:2],
                   self.__output_nodes_len, self.__output_features)
        return x[..., :3], rotation_6d_to_matrix(x[..., 3:])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, cooldown=10),
            'interval': 'epoch',
            'monitor': 'train_loss/primary'
        }

        config = {
            'optimizer': optimizer,
            # 'lr_scheduler': lr_scheduler,
        }

        return config
