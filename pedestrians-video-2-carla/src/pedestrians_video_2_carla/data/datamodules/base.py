from typing import Optional
from pedestrians_video_2_carla.data import OUTPUTS_BASE
import os
import hashlib
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pedestrians_video_2_carla.skeletons.nodes import Skeleton
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
import yaml

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper


class BaseDataModule(LightningDataModule):
    def __init__(self,
                 outputs_dir: Optional[str] = None,
                 clip_length: Optional[int] = 30,
                 batch_size: Optional[int] = 64,
                 num_workers: Optional[int] = os.cpu_count(),
                 input_nodes: Optional[Skeleton] = CARLA_SKELETON,
                 **kwargs):
        super().__init__()

        if outputs_dir is None:
            outputs_dir = os.path.join(OUTPUTS_BASE, self.__class__.__name__)
        self.outputs_dir = outputs_dir

        self.clip_length = clip_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nodes = input_nodes

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.transform = self._setup_data_transform()

        self._settings_digest = self._calculate_settings_digest()
        self._subsets_dir = os.path.join(
            self.outputs_dir, 'subsets', self._settings_digest)

        self._needs_preparation = False
        if not os.path.exists(self._subsets_dir):
            self._needs_preparation = True
            os.makedirs(self._subsets_dir)

        # TODO: add self.transform repr to hyperparams
        self.save_hyperparameters({
            **self.settings,
            'settings_digest': self._settings_digest
        })

    @property
    def settings(self):
        return {
            'data_module_name': self.__class__.__name__,
            'batch_size': self.batch_size,
            'clip_length': self.clip_length,
            'nodes': self.nodes.__name__,
        }

    def _calculate_settings_digest(self):
        return hashlib.md5('-'.join(['{}={}'.format(k, str(s))
                                     for k, s in self.settings.items()]).encode()).hexdigest()

    def save_settings(self):
        with open(os.path.join(self._subsets_dir, 'dparams.yaml'), 'w') as f:
            yaml.dump(self.settings, f, Dumper=Dumper)

    def _setup_data_transform(self):
        return None

    @ staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Base DataModule')
        parser.add_argument(
            "--outputs_dir",
            type=str,
            default=None,
            help="Output directory for the dataset. Defaults to {}/NameOfTheDataModuleClass".format(
                OUTPUTS_BASE)
        )
        parser.add_argument(
            "--clip_length",
            metavar='NUM_FRAMES',
            type=int,
            default=30,
            help="Length of the clips."
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="Batch size."
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=os.cpu_count(),
            help="Number of workers for the data loader."
        )
        # input nodes are handled in the model hyperparameters
        return parent_parser

    def _dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle
        )

    def train_dataloader(self):
        return self._dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_set)

    def test_dataloader(self):
        return self._dataloader(self.test_set)
