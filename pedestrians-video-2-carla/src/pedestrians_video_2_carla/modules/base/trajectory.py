from typing import Dict, List, Tuple
import torch
from torch import nn

from pedestrians_video_2_carla.modules.base.output_types import TrajectoryModelOutputType


class TrajectoryModel(nn.Module):
    """
    Base model for trajectory prediction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._hparams = {}

    @property
    def hparams(self):
        return {
            'trajectory_model_name': self.__class__.__name__,
            'trajectory_output_type': self.output_type.name,
            **self._hparams
        }

    @property
    def output_type(self):
        return TrajectoryModelOutputType.changes

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
