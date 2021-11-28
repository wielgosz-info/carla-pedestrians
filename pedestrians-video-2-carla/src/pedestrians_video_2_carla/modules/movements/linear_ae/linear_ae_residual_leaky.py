from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
from pedestrians_video_2_carla.modules.base.output_types import MovementsModelOutputType
import torch
from torch import nn
from pedestrians_video_2_carla.modules.base.movements import MovementsModel


class LinearAEResidualLeaky(MovementsModel):
    """
    Residual bottleneck autoencoder with LeakyReLU.
    Inputs are flattened to a vector of size (input_nodes_len * input_features).
    """

    def __init__(self,
                 linear_size=256,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 2  # (x, y)

        self.__output_nodes_len = len(self.output_nodes)
        self.__output_features = 9

        self.__input_size = self.__input_nodes_len * self.__input_features
        self.__output_size = self.__output_nodes_len * self.__output_features

        self.__encoder = nn.Sequential(
            nn.Linear(self.__input_size, linear_size),
            nn.Linear(linear_size, linear_size // 2),
            nn.BatchNorm1d(linear_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(linear_size // 2, linear_size // 4),
            nn.BatchNorm1d(linear_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(linear_size // 4, linear_size // 8),
            nn.BatchNorm1d(linear_size // 8),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )

        self.__residual_bottleneck = nn.Sequential(
            nn.Linear(self.__input_size, linear_size // 8),
            nn.BatchNorm1d(linear_size // 8),
            nn.LeakyReLU(),
        )

        self.__decoder = nn.Sequential(
            nn.Linear(linear_size // 8, linear_size // 4),
            nn.BatchNorm1d(linear_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(linear_size // 4, linear_size // 2),
            nn.BatchNorm1d(linear_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(linear_size // 2, linear_size),
            nn.Linear(linear_size, self.__output_size)
        )

        self._hparams = {
            'linear_size': linear_size,
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
        parser = parent_parser.add_argument_group("LinearAEResidual Lightning Module")
        parser.add_argument(
            '--linear_size',
            default=256,
            type=int,
        )
        return parent_parser

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        x = x.view((-1, self.__input_size))

        bottleneck = self.__encoder(x)
        bottleneck = bottleneck + self.__residual_bottleneck(x)
        x = self.__decoder(bottleneck)

        x = x.view(*original_shape[0:2],
                   self.__output_nodes_len, self.__output_features)
        return x[..., :3], rotation_6d_to_matrix(x[..., 3:])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        config = {
            'optimizer': optimizer,
        }

        return config
