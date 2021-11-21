from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
import torch
from torch import nn
from pedestrians_video_2_carla.modules.base.movements import MovementsModel

from pedestrians_video_2_carla.modules.base.output_types import MovementsModelOutputType


class Linear(MovementsModel):
    """
    The simplest dummy model used to debug the flow.
    """

    def __init__(self,
                 movements_output_type: MovementsModelOutputType = MovementsModelOutputType.pose_changes,
                 needs_confidence: bool = False,
                 **kwargs
                 ):
        super().__init__(
            **kwargs
        )

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 3 if needs_confidence else 2

        self.__output_nodes_len = len(self.output_nodes)
        self.__movements_output_type = movements_output_type
        self.__needs_confidence = needs_confidence

        if self.__movements_output_type == MovementsModelOutputType.pose_changes:
            self.__output_features = 6  # rotation 6D
            self.__transform = lambda x: rotation_6d_to_matrix(x)
        elif self.__movements_output_type == MovementsModelOutputType.absolute_loc:
            self.__output_features = 3  # x,y,z
            self.__transform = lambda x: x
        elif self.__movements_output_type == MovementsModelOutputType.absolute_loc_rot:
            self.__output_features = 9  # x,y,z + rotation 6D
            self.__transform = lambda x: (x[..., :3], rotation_6d_to_matrix(x[..., 3:]))

        self.__input_size = self.__input_nodes_len * self.__input_features
        self.__output_size = self.__output_nodes_len * self.__output_features

        self.linear = nn.Linear(
            self.__input_size,
            self.__output_size
        )

    @property
    def output_type(self) -> MovementsModelOutputType:
        return self.__movements_output_type

    @property
    def needs_confidence(self) -> bool:
        return self.__needs_confidence

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Linear Lightning Module")
        parser.add_argument(
            '--movements_output_type',
            help="""
                Set projection type to use.
                """.format(
                set(MovementsModelOutputType.__members__.keys())),
            default=MovementsModelOutputType.pose_changes,
            choices=list(MovementsModelOutputType),
            type=MovementsModelOutputType.__getitem__
        )
        parser.add_argument(
            '--needs_confidence',
            dest='needs_confidence',
            action='store_true',
            default=False,
        )
        return parent_parser

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        x = x.view((-1, self.__input_size))
        pose_change = self.linear(x)
        pose_change = pose_change.view(*original_shape[0:2],
                                       self.__output_nodes_len, self.__output_features)

        return self.__transform(pose_change)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        config = {
            'optimizer': optimizer,
        }

        return config
