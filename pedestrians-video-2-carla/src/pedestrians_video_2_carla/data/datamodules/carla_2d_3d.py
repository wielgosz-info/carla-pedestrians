from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from pedestrians_video_2_carla.data.datamodules.base import BaseDataModule
from pedestrians_video_2_carla.data.datasets.carla_2d_3d_dataset import Carla2D3DDataset, Carla2D3DIterableDataset
import os
import hashlib
import h5py
import numpy as np
from pedestrians_video_2_carla.data import OUTPUTS_BASE

OUTPUTS_DIR = os.path.join(OUTPUTS_BASE, 'JAAD')


class Carla2D3DDataModule(BaseDataModule):
    def __init__(self,
                 random_changes_each_frame: Optional[int] = 3,
                 max_change_in_deg: Optional[int] = 5,
                 **kwargs):
        self.random_changes_each_frame = random_changes_each_frame
        self.max_change_in_deg = max_change_in_deg

        super().__init__(**kwargs)

        self.save_hyperparameters({
            'random_changes_each_frame': self.random_changes_each_frame,
            'max_change_in_deg': self.max_change_in_deg
        })

    def _calculate_settings_digest(self):
        return hashlib.md5(''.join([str(s) for s in [
            self.clip_length,
            self.random_changes_each_frame,
            self.max_change_in_deg
        ]]).encode()).hexdigest()

    @staticmethod
    def add_data_specific_args(parent_parser):
        BaseDataModule.add_data_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("Carla2D3D DataModule")
        parser.add_argument("--random_changes_each_frame", type=int, default=3,
                            help="Number of nodes that will be randomly changed in each frame")
        parser.add_argument("--max_change_in_deg", type=int, default=5,
                            help="Max random change in degrees")
        return parent_parser

    def prepare_data(self) -> None:
        if not self._needs_preparation:
            return

        # generate and save validation & test sets so they are reproducible
        iterable_dataset = Carla2D3DIterableDataset(
            clip_length=self.clip_length,
            nodes=self.nodes,
            transform=self.transform,
            random_changes_each_frame=self.random_changes_each_frame,
            max_change_in_deg=self.max_change_in_deg
        )

        # for now, we generate 2 validation batches and 3 test batches
        val_set_size = 2 * self.batch_size
        test_set_size = 3 * self.batch_size

        sizes = [val_set_size, test_set_size]
        names = ['val', 'test']
        for (size, name) in zip(sizes, names):
            clips_set = tuple(zip(*[next(iter(iterable_dataset)) for _ in range(size)]))
            projection_2d = np.stack(clips_set[0], axis=0)
            pose_changes = np.stack(clips_set[1], axis=0)
            meta = {k: [dic[k] for dic in clips_set[2]] for k in clips_set[2][0]}

            with h5py.File(os.path.join(self._subsets_dir, "{}.hdf5".format(name)), "w") as f:
                f.create_dataset("carla_2d_3d/projection_2d", data=projection_2d,
                                 chunks=(1, *projection_2d.shape[1:]))
                f.create_dataset("carla_2d_3d/pose_changes", data=pose_changes,
                                 chunks=(1, *pose_changes.shape[1:]))
                for k, v in meta.items():
                    unique = list(set(v))
                    labels = np.array([
                        s.encode("latin-1") for s in unique
                    ], dtype=h5py.string_dtype('ascii', 30))
                    mapping = {s: i for i, s in enumerate(unique)}
                    f.create_dataset("carla_2d_3d/meta/{}".format(k),
                                     data=[mapping[s] for s in v], dtype=np.uint16)
                    f["carla_2d_3d/meta/{}".format(k)].attrs["labels"] = labels

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_set = Carla2D3DIterableDataset(
                clip_length=self.clip_length,
                nodes=self.nodes,
                transform=self.transform,
                random_changes_each_frame=self.random_changes_each_frame,
                max_change_in_deg=self.max_change_in_deg
            )
            self.val_set = Carla2D3DDataset(
                os.path.join(self._subsets_dir, 'val.hdf5'),
                nodes=self.nodes,
                transform=self.transform
            )

        if stage == "test" or stage is None:
            self.test_set = Carla2D3DDataset(
                os.path.join(self._subsets_dir, 'test.hdf5'),
                nodes=self.nodes,
                transform=self.transform
            )

    def __dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            shuffle=shuffle
        )

    def train_dataloader(self):
        # no need to shuffle, it is randomly generated
        return self.__dataloader(self.train_set)

    def val_dataloader(self):
        return self.__dataloader(self.val_set)

    def test_dataloader(self):
        return self.__dataloader(self.test_set)
