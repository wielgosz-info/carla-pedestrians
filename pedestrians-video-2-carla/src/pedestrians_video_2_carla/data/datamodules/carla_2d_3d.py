from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from pedestrians_video_2_carla.data.datasets.carla_2d_3d_dataset import Carla2D3DDataset
from pedestrians_video_2_carla.transforms.hips_neck import CarlaHipsNeckNormalize
from pedestrians_video_2_carla.skeletons.points.carla import CARLA_SKELETON
import os


class Carla2D3DDataModule(LightningDataModule):
    def __init__(self,
                 clip_length: Optional[int] = 30,
                 batch_size: Optional[int] = 64,
                 points: Optional[CARLA_SKELETON] = CARLA_SKELETON,
                 **kwargs):
        super().__init__()
        self.clip_length = clip_length
        self.batch_size = batch_size
        self.points = points

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_set = Carla2D3DDataset(
                self.data_dir,
                os.path.join(self.__subsets_dir, 'train.csv'),
                points=self.points,
                transform=CarlaHipsNeckNormalize(self.points)
            )
            self.val_set = Carla2D3DDataset(
                self.data_dir,
                os.path.join(self.__subsets_dir, 'val.csv'),
                points=self.points,
                transform=CarlaHipsNeckNormalize(self.points)
            )

        if stage == "test" or stage is None:
            self.test_set = Carla2D3DDataset(
                self.data_dir,
                os.path.join(self.__subsets_dir, 'test.csv'),
                points=self.points,
                transform=CarlaHipsNeckNormalize(self.points)
            )

    def __dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=0,  # os.cpu_count(),
            pin_memory=True,
            shuffle=shuffle
        )

    def train_dataloader(self):
        return self.__dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.val_set)

    def test_dataloader(self):
        return self.__dataloader(self.test_set)
