import torch
from torch import nn
from pedestrians_video_2_carla.modules.base.movements import MovementsModel

from pedestrians_video_2_carla.modules.base.output_types import MovementsModelOutputType
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix


class LinearAEResidual(MovementsModel):
    """
    Residual bottleneck autoencoder with ReLU.
    Inputs are flattened to a vector of size (input_nodes_len * input_features).
    """

    def __init__(self,
                 linear_size=256,
                 activation_cls=nn.ReLU,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._input_nodes_len = len(self.input_nodes)
        self._input_features = 2  # (x, y)

        self._output_nodes_len = len(self.output_nodes)
        self._output_features = 9

        self._input_size = self._input_nodes_len * self._input_features
        self._output_size = self._output_nodes_len * self._output_features

        self._encoder = nn.Sequential(
            nn.Linear(self._input_size, linear_size),
            nn.Linear(linear_size, linear_size // 2),
            nn.BatchNorm1d(linear_size // 2),
            activation_cls(),
            nn.Dropout(0.5),
            nn.Linear(linear_size // 2, linear_size // 4),
            nn.BatchNorm1d(linear_size // 4),
            activation_cls(),
            nn.Dropout(0.5),
            nn.Linear(linear_size // 4, linear_size // 8),
            nn.BatchNorm1d(linear_size // 8),
            activation_cls(),
            nn.Dropout(0.5)
        )

        self._residual_bottleneck = nn.Sequential(
            nn.Linear(self._input_size, linear_size // 8),
            nn.BatchNorm1d(linear_size // 8),
            activation_cls(),
        )

        self._decoder = nn.Sequential(
            nn.Linear(linear_size // 8, linear_size // 4),
            nn.BatchNorm1d(linear_size // 4),
            activation_cls(),
            nn.Dropout(0.5),
            nn.Linear(linear_size // 4, linear_size // 2),
            nn.BatchNorm1d(linear_size // 2),
            activation_cls(),
            nn.Dropout(0.5),
            nn.Linear(linear_size // 2, linear_size),
            nn.Linear(linear_size, self._output_size),
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
        x = x.view((-1, self._input_size))

        bottleneck = self._encoder(x)
        bottleneck = bottleneck + self._residual_bottleneck(x)
        x = self._decoder(bottleneck)

        x = x.view(*original_shape[0:2],
                   self._output_nodes_len, self._output_features)
        return x[..., :3], rotation_6d_to_matrix(x[..., 3:])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        config = {
            'optimizer': optimizer,
        }

        return config
