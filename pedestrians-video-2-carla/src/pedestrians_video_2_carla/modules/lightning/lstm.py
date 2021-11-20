from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
import torch
from torch import nn

from pedestrians_video_2_carla.modules.lightning.base import LitBaseMapper
from pedestrians_video_2_carla.modules.projection.projection import ProjectionTypes


class LSTM(LitBaseMapper):
    """
    Very basic Linear + LSTM + Linear model.
    """

    def __init__(self,
                 projection_type: ProjectionTypes = ProjectionTypes.pose_changes,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 **kwargs
                 ):
        super().__init__(projection_type=projection_type, **kwargs,)

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 2  # (x, y) points

        self.__output_nodes_len = len(self.output_nodes)
        if projection_type == ProjectionTypes.pose_changes:
            self.__output_features = 6  # rotation 6D
            self.__transform = lambda x: rotation_6d_to_matrix(x)
        elif projection_type == ProjectionTypes.absolute_loc:
            self.__output_features = 3  # x,y,z
            self.__transform = lambda x: x
        elif projection_type == ProjectionTypes.absolute_loc_rot:
            self.__output_features = 9  # x,y,z + rotation 6D
            self.__transform = lambda x: (x[..., :3], rotation_6d_to_matrix(x[..., 3:]))

        self.__input_size = self.__input_nodes_len * self.__input_features
        self.__output_size = self.__output_nodes_len * self.__output_features

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

        self.save_hyperparameters({
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'projection_type': projection_type.name
        })

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = LitBaseMapper.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("Linear Lightning Module")
        parser.add_argument(
            '--projection_type',
            help="""
                Set projection type to use.
                """.format(
                set(ProjectionTypes.__members__.keys())),
            default=ProjectionTypes.pose_changes,
            choices=list(ProjectionTypes),
            type=ProjectionTypes.__getitem__
        )
        return parent_parser

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        x = x.view(*original_shape[0:2], self.__input_size)
        x = self.linear_1(x)
        x, _ = self.lstm_1(x)
        pose_change = self.linear_2(x)
        pose_change = pose_change.view(*original_shape[0:2],
                                       self.__output_nodes_len, self.__output_features)
        return self.__transform(pose_change)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        return optimizer
