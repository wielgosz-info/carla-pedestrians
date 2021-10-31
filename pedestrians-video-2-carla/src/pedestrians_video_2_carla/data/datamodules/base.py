from typing import Callable, Optional
from pedestrians_video_2_carla.data import OUTPUTS_BASE
import os
from pytorch_lightning import LightningDataModule

from pedestrians_video_2_carla.skeletons.nodes import Skeleton
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON


OUTPUTS_DIR = os.path.join(OUTPUTS_BASE, 'JAAD')


class BaseDataModule(LightningDataModule):
    def __init__(self,
                 outputs_dir: Optional[str] = OUTPUTS_DIR,
                 clip_length: Optional[int] = 30,
                 batch_size: Optional[int] = 64,
                 input_nodes: Optional[Skeleton] = CARLA_SKELETON,
                 transform: Optional[Callable] = None,
                 **kwargs):
        super().__init__()

        self.outputs_dir = outputs_dir
        self.clip_length = clip_length
        self.batch_size = batch_size
        self.nodes = input_nodes

        if transform is not None:
            self.transform = transform(self.nodes)
        else:
            self.transform = None

        self._settings_digest = self._calculate_settings_digest()
        self._subsets_dir = os.path.join(
            self.outputs_dir, 'subsets', self._settings_digest)

        self._needs_preparation = False
        if not os.path.exists(self._subsets_dir):
            self._needs_preparation = True
            os.makedirs(self._subsets_dir)

        # TODO: add self.transform repr to hyperparams
        self.save_hyperparameters({
            'batch_size': self.batch_size,
            'clip_length': self.clip_length,
            'settings_digest': self._settings_digest
        })

    def _calculate_settings_digest(self):
        raise NotImplementedError()

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Base DataModule')
        parser.add_argument("--outputs_dir", type=str, default=OUTPUTS_DIR,
                            help="Output directory for the dataset")
        parser.add_argument("--clip_length", type=int, default=30,
                            help="Length of the clips")
        parser.add_argument("--batch_size", type=int, default=64,
                            help="Batch size")
        # input nodes are handled in the model hyperparameters
        return parent_parser
