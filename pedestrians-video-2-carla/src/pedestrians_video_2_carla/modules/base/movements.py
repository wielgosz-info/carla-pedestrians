from typing import Dict, List, Tuple, Type
import torch
from torch import nn
from pedestrians_video_2_carla.modules.base.output_types import MovementsModelOutputType
from pedestrians_video_2_carla.skeletons.nodes import Skeleton, get_skeleton_name_by_type
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON


class MovementsModel(nn.Module):
    """
    Base interface for movement models.
    """

    def __init__(self,
                 input_nodes: Type[Skeleton] = CARLA_SKELETON,
                 output_nodes: Type[Skeleton] = CARLA_SKELETON,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self._hparams = {}

    @property
    def hparams(self):
        return {
            'movements_model_name': self.__class__.__name__,
            'movements_output_type': self.output_type.name,
            'input_nodes': get_skeleton_name_by_type(self.input_nodes),
            'output_nodes': get_skeleton_name_by_type(self.output_nodes),
            **self._hparams
        }

    @property
    def output_type(self) -> MovementsModelOutputType:
        return MovementsModelOutputType.pose_changes

    @property
    def needs_confidence(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model-specific arguments to the CLI args parser.
        """
        return parent_parser

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, '_LRScheduler']]]:
        raise NotImplementedError()

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError()
