from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from pedestrians_video_2_carla.data.datasets.carla_2d_3d_dataset import Carla2D3DDataset, Carla2D3DIterableDataset
from pedestrians_video_2_carla.transforms.hips_neck import CarlaHipsNeckNormalize
from pedestrians_video_2_carla.skeletons.points.carla import CARLA_SKELETON
import os
import hashlib
import h5py
import numpy as np
from pedestrians_video_2_carla.data import OUTPUTS_BASE

OUTPUTS_DIR = os.path.join(OUTPUTS_BASE, 'JAAD')


class Carla2D3DDataModule(LightningDataModule):
    def __init__(self,
                 outputs_dir: Optional[str] = OUTPUTS_DIR,
                 clip_length: Optional[int] = 30,
                 batch_size: Optional[int] = 64,
                 points: Optional[CARLA_SKELETON] = CARLA_SKELETON,
                 **kwargs):
        super().__init__()
        self.outputs_dir = outputs_dir
        self.clip_length = clip_length
        self.batch_size = batch_size
        self.points = points

        self.__settings_digest = hashlib.md5(
            (str(clip_length)+str(batch_size)).encode()).hexdigest()
        self.__subsets_dir = os.path.join(
            self.outputs_dir, 'subsets', self.__settings_digest)

        self.__needs_preparation = False
        if not os.path.exists(self.__subsets_dir):
            self.__needs_preparation = True
            os.makedirs(self.__subsets_dir)

    def prepare_data(self) -> None:
        if not self.__needs_preparation:
            return

        # generate and save validation & test sets so they are reproducible
        iterable_dataset = Carla2D3DIterableDataset(
            clip_length=self.clip_length,
            points=self.points,
            transform=CarlaHipsNeckNormalize(self.points)
        )

        # for now, we generate 2 validation batches and 3 test batches
        val_set_size = 2 * self.batch_size
        test_set_size = 3 * self.batch_size

        sizes = [val_set_size, test_set_size]
        names = ['val', 'test']
        for (size, name) in zip(sizes, names):
            clips_set = tuple(zip(*[next(iter(iterable_dataset)) for _ in range(size)]))
            projection_2d = np.concatenate(clips_set[0], axis=0)
            pose_changes = np.concatenate(clips_set[1], axis=0)
            meta = {k: [dic[k] for dic in clips_set[2]] for k in clips_set[2][0]}

            with h5py.File(os.path.join(self.__subsets_dir, "{}.hdf5".format(name)), "w") as f:
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
                points=self.points,
                transform=CarlaHipsNeckNormalize(self.points)
            )
            self.val_set = Carla2D3DDataset(
                os.path.join(self.__subsets_dir, 'val.hdf5'),
                points=self.points,
                transform=CarlaHipsNeckNormalize(self.points)
            )

        if stage == "test" or stage is None:
            self.test_set = Carla2D3DDataset(
                os.path.join(self.__subsets_dir, 'test.hdf5'),
                points=self.points,
                transform=CarlaHipsNeckNormalize(self.points)
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
