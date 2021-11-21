import os
from typing import Optional

import h5py
import numpy as np
from pytorch_lightning.utilities.warnings import rank_zero_warn
import torch
from pedestrians_video_2_carla.data.datamodules.base import BaseDataModule
from pedestrians_video_2_carla.data.datasets.carla_2d_3d_dataset import (
    Carla2D3DDataset, Carla2D3DIterableDataset)
from tqdm import trange


class Carla2D3DDataModule(BaseDataModule):
    def __init__(self,
                 val_batches: Optional[int] = 2,
                 test_batches: Optional[int] = 3,
                 **kwargs):
        self.val_batches = val_batches
        self.test_batches = test_batches
        self.kwargs = kwargs

        super().__init__(**kwargs)

        if kwargs.get("limit_train_batches", None) is None:
            rank_zero_warn(
                "No limit on train batches was set (--limit_train_batches), this will result in infinite training.")

    @property
    def settings(self):
        return {
            **super().settings,
            'random_changes_each_frame': self.kwargs.get('random_changes_each_frame'),
            'max_change_in_deg': self.kwargs.get('max_change_in_deg'),
            'max_world_rot_change_in_deg': self.kwargs.get('max_world_rot_change_in_deg'),
            'max_root_yaw_change_in_deg': self.kwargs.get('max_root_yaw_change_in_deg'),
            'missing_point_probability': self.kwargs.get('missing_point_probability'),
            'val_set_size': self.val_batches * self.batch_size,
            'test_set_size': self.test_batches * self.batch_size,
        }

    @staticmethod
    def add_data_specific_args(parent_parser):
        BaseDataModule.add_data_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("Carla2D3D DataModule")
        parser.add_argument(
            "--random_changes_each_frame",
            type=int,
            default=3,
            metavar='NUM_NODES',
            help="Number of nodes that will be randomly changed in each frame."
        )
        parser.add_argument(
            "--max_change_in_deg",
            type=int,
            default=5,
            metavar='DEGREES',
            help="Max random [+/-] change in degrees."
        )
        parser.add_argument(
            "--max_world_rot_change_in_deg",
            type=int,
            default=0,
            metavar='DEGREES',
            help="Max random [+/-] world rotation yaw change in degrees. IMPLEMENTATION IN PROGRESS, DO NOT USE."
        )
        parser.add_argument(
            "--max_root_yaw_change_in_deg",
            type=int,
            default=0,
            metavar='DEGREES',
            help="Max random [+/-] 'world' rotation yaw change in degrees. TEMPORARY IMPLEMENTATION."
        )
        parser.add_argument(
            "--missing_point_probability",
            type=float,
            default=0.0,
            metavar='PROB',
            help="""
                Probability that a joint/node will be missing ("not detected") in a skeleton in a frame.
                Missing nodes are selected separately for each frame.
            """
        )
        return parent_parser

    def prepare_data(self) -> None:
        if not self._needs_preparation:
            return

        # generate and save validation & test sets so they are reproducible
        iterable_dataset = Carla2D3DIterableDataset(
            nodes=self.nodes,
            **{
                **self.kwargs,
                'transform': None  # we want raw data in dataset
            },
        )

        sizes = [self.val_batches, self.test_batches]
        names = ['val', 'test']
        for (size, name) in zip(sizes, names):
            if size <= 0:
                continue

            clips_set = tuple(zip(*[iterable_dataset.generate_batch()
                              for _ in trange(size, desc=f'Generating {name} set')]))
            projection_2d = torch.cat(clips_set[0], dim=0).cpu().numpy()
            targets = {k: [dic[k] for dic in clips_set[1]] for k in clips_set[1][0]}
            meta = {k: [dic[k] for dic in clips_set[2]] for k in clips_set[2][0]}

            with h5py.File(os.path.join(self._subsets_dir, "{}.hdf5".format(name)), "w") as f:
                f.create_dataset("carla_2d_3d/projection_2d", data=projection_2d,
                                 chunks=(1, *projection_2d.shape[1:]))

                for k, v in targets.items():
                    stacked_v = np.concatenate(v, axis=0)
                    f.create_dataset(f"carla_2d_3d/targets/{k}", data=stacked_v,
                                     chunks=(1, *stacked_v.shape[1:]))

                for k, v in meta.items():
                    stacked_v = np.concatenate(v, axis=0)
                    unique = list(set(stacked_v))
                    labels = np.array([
                        s.encode("latin-1") for s in unique
                    ], dtype=h5py.string_dtype('ascii', 30))
                    mapping = {s: i for i, s in enumerate(unique)}
                    f.create_dataset("carla_2d_3d/meta/{}".format(k),
                                     data=[mapping[s] for s in stacked_v], dtype=np.uint16)
                    f["carla_2d_3d/meta/{}".format(k)].attrs["labels"] = labels

        # save settings
        self.save_settings()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_set = Carla2D3DIterableDataset(
                nodes=self.nodes,
                transform=self.transform,
                **self.kwargs,
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

    def train_dataloader(self):
        # no need to shuffle, it is randomly generated
        return self._dataloader(self.train_set)
