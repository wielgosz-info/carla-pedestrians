import os
import pandas
from typing import Dict, Optional
from pandas.core.frame import DataFrame
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule

DATA_DIR = os.path.join('outputs', 'JAAD')
DF_ISIN = {
    'action': ['walking'],
    'speed': ['stopped'],
    'group_size': [1]
}
DF_USECOLS = [
    'video',
    'frame',
    'x1',
    'y1',
    'x2',
    'y2',
    'id',
    'action',
    'gender',
    'age',
    'speed',
    'group_size'
]


class JAADOpenPoseDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: Optional[str] = DATA_DIR,
                 df_usecols=DF_USECOLS,
                 df_isin: Optional[Dict] = DF_ISIN,
                 clip_length: Optional[int] = 30,
                 clip_offset: Optional[int] = 10):
        super().__init__()

        self.data_dir = data_dir
        self.annotations_filepath = os.path.join(self.data_dir, 'annotations.csv')
        self.annotations_usecols = df_usecols
        self.annotations_filters = df_isin

        self.clip_length = clip_length
        self.clip_offset = clip_offset

    def prepare_data(self) -> None:
        # this is only called on one GPU, do not use self.something assignments
        annotations_df: DataFrame = pandas.read_csv(
            self.annotations_filepath,
            usecols=self.annotations_usecols,
            index_col=['video', 'frame', 'id']
        )
        filtering_results = annotations_df.isin(self.annotations_filters)[
            list(self.annotations_filters.keys())].all(axis=1)

        annotations_df = annotations_df[filtering_results]
        video_groupby = annotations_df.groupby(['video', 'id'])

        frame_counts = video_groupby['age'].count()
        has_min_frame_count = frame_counts[frame_counts >= self.clip_length].index

        # TODO: we need clips with continuous frames, offset from each other's starting position by at least clip_offset

    # def setup(self, stage: Optional[str] = None) -> None:

    #     if stage == "fit" or stage is None:
    #         # prepare train and val

    #     if stage == "test" or stage is None:
    #         # prepare test
