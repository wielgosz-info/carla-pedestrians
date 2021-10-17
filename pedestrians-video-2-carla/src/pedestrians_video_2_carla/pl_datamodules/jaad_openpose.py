import os
import pandas
import math
import uuid
from typing import Dict, Optional
from pandas.core.frame import DataFrame
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule

DATA_DIR = os.path.join('/outputs', 'JAAD')
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

        if not os.path.exists(os.path.join(self.data_dir, 'subsets')):
            os.mkdir(os.path.join(self.data_dir, 'subsets'))
        self.subsets_dir = os.path.join(
            self.data_dir, 'subsets', str(uuid.uuid4()))
        os.mkdir(self.subsets_dir)

    def prepare_data(self) -> None:
        # this is only called on one GPU, do not use self.something assignments
        annotations_df: DataFrame = pandas.read_csv(
            self.annotations_filepath,
            usecols=self.annotations_usecols,
            index_col=['video', 'id']
        )

        filtering_results = annotations_df.isin(self.annotations_filters)[
            list(self.annotations_filters.keys())].all(axis=1)

        annotations_df = annotations_df[filtering_results]
        annotations_df.sort_index(inplace=True)

        frame_counts = annotations_df.groupby(['video', 'id']).agg(
            frame_count=pandas.NamedAgg(column="frame", aggfunc="count"),
            frame_min=pandas.NamedAgg(column="frame", aggfunc="min"),
            frame_max=pandas.NamedAgg(column="frame", aggfunc="max")
        ).assign(
            frame_diff=lambda x: x.frame_max - x.frame_min + 1
        ).assign(frame_count_eq_diff=lambda x: x.frame_count == x.frame_diff)

        # drop clips that are too short for sure
        frame_counts = frame_counts[frame_counts.frame_count >= self.clip_length]

        clips = DataFrame([], columns=DF_USECOLS+['clip']
                          ).set_index(['video', 'id', 'clip', 'frame'])

        # handle continuous clips first
        for idx in frame_counts[frame_counts.frame_count_eq_diff == True].index:
            video = annotations_df.loc[idx]
            ci = 0
            while (ci*self.clip_offset + self.clip_length) <= frame_counts.loc[idx].frame_count:
                clips = clips.append(video.iloc[ci * self.clip_offset:ci *
                                                self.clip_offset + self.clip_length].reset_index()[self.annotations_usecols].assign(clip=ci).set_index(['video', 'id', 'clip', 'frame']))
                ci += 1

        # then try to extract from non-continuos
        for idx in frame_counts[frame_counts.frame_count_eq_diff == False].index:
            video = annotations_df.loc[idx]
            frame_diffs_min = video[1:][['frame']].assign(
                frame_diff=video[1:].frame - video[0:-1].frame)
            frame_min = [frame_counts.loc[idx].frame_min] + \
                list(frame_diffs_min[frame_diffs_min.frame_diff > 1]['frame'].values)
            frame_diffs_max = video[0:-1][['frame']
                                          ].assign(frame_diff=video[1:].frame - video[0:-1].frame)
            frame_max = list(frame_diffs_max[frame_diffs_max.frame_diff > 1]
                             ['frame'].values) + [frame_counts.loc[idx].frame_max]
            ci = 0  # continuous for all clips
            for (fmin, fmax) in zip(frame_min, frame_max):
                while (ci*self.clip_offset + self.clip_length + fmin) <= fmax:
                    clips = clips.append(video.loc[video.frame >= ci*self.clip_offset + self.clip_length + fmin][:self.clip_length].reset_index()[
                                         self.annotations_usecols].assign(clip=ci).set_index(['video', 'id', 'clip', 'frame']))
                    ci += 1

        # aaaand finally we have what we need in "clips" to create our dataset
        # how many clips do we have?
        clip_counts = clips.reset_index(level=['clip', 'frame']).groupby(['video', 'id']).agg(
            clips_count=pandas.NamedAgg(column='clip', aggfunc='nunique')).sort_values('clips_count', ascending=False)
        clip_counts = clip_counts.assign(clips_cumsum=clip_counts.cumsum())
        total = clip_counts['clips_count'].sum()

        test_count = math.floor(total*0.2)
        val_count = math.floor((total-test_count)*0.2)
        train_count = total - test_count - val_count

        # we do not want to assign clips from the same video/pedestrian combination to different datasets,
        # especially since they are overlapping by default
        # so we try to assign them in roundrobin fashion
        # start by assigning the videos with most clips

        targets = (train_count, val_count, test_count)
        sets = [[], [], []]  # train, val, test
        current = [0, 0, 0]
        assigned = 0

        while assigned < total:
            skipped = 0
            for i in range(3):
                needed = targets[i] - current[i]
                if needed > 0:
                    to_assign = clip_counts[(assigned < clip_counts['clips_cumsum']) &
                                            (clip_counts['clips_cumsum'] <= assigned+needed)]
                    if not len(to_assign):
                        skipped += 1
                        continue
                    current[i] += to_assign['clips_count'].sum()
                    sets[i].append(to_assign)
                    assigned = sum(current)
            if skipped == 3:
                # assign whatever is left to train set
                sets[0].append(clip_counts[assigned < clip_counts['clips_cumsum']])
                break

        # now we need to dump the actual clips info
        names = ['train', 'val', 'test']
        for (i, name) in enumerate(names):
            clips_set = clips.join(pandas.concat(sets[i]), how='right')
            clips_set.drop(['clips_count', 'clips_cumsum'], inplace=True, axis=1)
            clips_set.to_csv(os.path.join(self.subsets_dir, '{:s}.csv'.format(name)))

    # def setup(self, stage: Optional[str] = None) -> None:

    #     if stage == "fit" or stage is None:
    #         # prepare train and val

    #     if stage == "test" or stage is None:
    #         # prepare test

    def train_dataloader(self):
        return DataLoader()
