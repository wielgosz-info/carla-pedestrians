import math
from typing import Callable, Optional, Type

import h5pickle as h5py
import numpy as np
import torch
from pedestrians_video_2_carla.modules.layers.projection import \
    ProjectionModule
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pytorch3d.transforms import euler_angles_to_matrix
from torch.functional import Tensor
from torch.utils.data import Dataset, IterableDataset


class Carla2D3DDataset(Dataset):
    def __init__(self, set_filepath: str, nodes: CARLA_SKELETON = CARLA_SKELETON, transform=None, **kwargs) -> None:
        set_file = h5py.File(set_filepath, 'r')

        self.projection_2d = set_file['carla_2d_3d/projection_2d']
        self.pose_changes = set_file['carla_2d_3d/targets/pose_changes']
        self.world_loc_changes = set_file['carla_2d_3d/targets/world_loc_changes']
        self.world_rot_changes = set_file['carla_2d_3d/targets/world_rot_changes']
        self.absolute_pose_loc = set_file['carla_2d_3d/targets/absolute_pose_loc']
        self.absolute_pose_rot = set_file['carla_2d_3d/targets/absolute_pose_rot']
        self.meta = set_file['carla_2d_3d/meta']

        self.transform = transform
        self.nodes = nodes

    def __len__(self) -> int:
        return len(self.projection_2d)

    def __getitem__(self, idx: int) -> torch.Tensor:
        projection_2d = self.projection_2d[idx]
        projection_2d = torch.tensor(projection_2d)
        if self.transform:
            projection_2d = self.transform(projection_2d)

        pose_changes_matrix = self.pose_changes[idx]
        pose_changes_matrix = torch.from_numpy(pose_changes_matrix)

        world_rot_change_batch = self.world_rot_changes[idx]
        world_rot_change_batch = torch.from_numpy(world_rot_change_batch)

        world_loc_change_batch = self.world_loc_changes[idx]
        world_loc_change_batch = torch.from_numpy(world_loc_change_batch)

        absolute_pose_loc = self.absolute_pose_loc[idx]
        absolute_pose_loc = torch.from_numpy(absolute_pose_loc)

        absolute_pose_rot = self.absolute_pose_rot[idx]
        absolute_pose_rot = torch.from_numpy(absolute_pose_rot)

        meta = {k: self.meta[k].attrs['labels'][v[idx]].decode(
            "latin-1") for k, v in self.meta.items()}

        return (
            projection_2d,
            {
                'pose_changes': pose_changes_matrix,
                'world_loc_changes': world_loc_change_batch,
                'world_rot_changes': world_rot_change_batch,
                'absolute_pose_loc': absolute_pose_loc,
                'absolute_pose_rot': absolute_pose_rot,
            },
            meta
        )


class Carla2D3DIterableDataset(IterableDataset):
    def __init__(self,
                 batch_size: Optional[int] = 64,
                 clip_length: Optional[int] = 30,
                 random_changes_each_frame: Optional[int] = 3,
                 max_change_in_deg: Optional[int] = 5,
                 max_world_rot_change_in_deg: Optional[int] = 0,
                 max_initial_world_rot_change_in_deg: Optional[int] = 0,
                 missing_point_probability: Optional[float] = 0.0,
                 nodes: Optional[Type[CARLA_SKELETON]] = CARLA_SKELETON,
                 transform: Optional[Callable[[Tensor], Tensor]] = None,
                 **kwargs) -> None:
        self.transform = transform
        self.nodes = nodes
        self.clip_length = clip_length
        self.random_changes_each_frame = random_changes_each_frame
        self.max_change_in_rad = np.deg2rad(max_change_in_deg)
        self.max_world_rot_change_in_rad = np.deg2rad(max_world_rot_change_in_deg)
        self.max_initial_world_rot_change_in_rad = np.deg2rad(
            max_initial_world_rot_change_in_deg)
        self.missing_point_probability = missing_point_probability
        self.batch_size = batch_size

        self.projection = ProjectionModule(
            input_nodes=self.nodes,
            output_nodes=self.nodes,
            projection_transform=lambda x: x,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
        else:
            num_workers = worker_info.num_workers

        bs = math.ceil(self.batch_size / num_workers)

        while True:
            inputs, targets, meta = self.generate_batch(bs)
            for idx in range(bs):
                yield (
                    inputs[idx],
                    {k: v[idx] for k, v in targets.items()},
                    {k: v[idx] for k, v in meta.items()}
                )

    def generate_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        nodes_size = len(self.nodes)
        nodes_nums = np.arange(nodes_size)
        pose_changes = torch.zeros(
            (batch_size, self.clip_length, nodes_size, 3))
        world_rot_change = torch.zeros(
            (batch_size, self.clip_length, 3))
        world_loc_change_batch = torch.zeros(
            (batch_size, self.clip_length, 3))

        for idx in range(batch_size):
            for i in range(self.clip_length):
                indices = np.random.choice(nodes_nums,
                                           size=self.random_changes_each_frame, replace=False)
                pose_changes[idx, i, indices] = (torch.rand(
                    (self.random_changes_each_frame, 3)) * 2 - 1) * self.max_change_in_rad

        pose_changes_batch = euler_angles_to_matrix(pose_changes, "XYZ")

        # only change yaw
        # TODO: for now, all initial rotations are equally probable
        if self.max_initial_world_rot_change_in_rad > 0:
            world_rot_change[:, 0, 2] = (torch.rand(
                (batch_size)) * 2 - 1) * self.max_initial_world_rot_change_in_rad
        # apply additional rotation changes during the clip
        if self.max_world_rot_change_in_rad != 0.0:
            world_rot_change[:, 1:, 2] = (torch.rand(
                (batch_size, self.clip_length-1)) * 2 - 1) * self.max_world_rot_change_in_rad
        world_rot_change_batch = euler_angles_to_matrix(world_rot_change, "XYZ")

        # TODO: introduce world location change at some point

        # TODO: we should probably take care of the "correct" pedestrians data distribution
        # need to find some pedestrian statistics
        age = np.random.choice(['adult', 'child'], size=batch_size)
        gender = np.random.choice(['male', 'female'], size=batch_size)

        self.projection.on_batch_start((pose_changes_batch, None, {
            'age': age,
            'gender': gender
        }), 0)
        projection_2d, absolute_pose_loc, absolute_pose_rot, *_ = self.projection.project_pose(
            pose_inputs_batch=pose_changes_batch,
            world_rot_change_batch=world_rot_change_batch,
            world_loc_change_batch=world_loc_change_batch,
        )

        # use the third dimension as 'confidence' of the projection
        # so we're compatible with OpenPose
        # this will also prevent the models from accidentally using
        # the depth data that pytorch3d leaves in the projections
        projection_2d[..., 2] = 1.0

        if self.missing_point_probability > 0.0:
            missing_indices = torch.rand(
                (batch_size, self.clip_length, nodes_size)) < self.missing_point_probability
            projection_2d[missing_indices] = torch.tensor(
                [0.0, 0.0, 0.0], device=projection_2d.device)

        if self.transform:
            projection_2d = self.transform(projection_2d)

        return (
            projection_2d,
            {
                'pose_changes': pose_changes_batch,
                'world_loc_changes': world_loc_change_batch,
                'world_rot_changes': world_rot_change_batch,
                'absolute_pose_loc': absolute_pose_loc,
                'absolute_pose_rot': absolute_pose_rot
            },
            {'age': age, 'gender': gender}
        )


if __name__ == "__main__":
    from pedestrians_video_2_carla.transforms.hips_neck import \
        HipsNeckNormalize
    from pedestrians_video_2_carla.utils.timing import print_timing, timing

    nodes = CARLA_SKELETON
    iter_dataset = Carla2D3DIterableDataset(
        batch_size=256,
        clip_length=180,
        transform=HipsNeckNormalize(nodes.get_extractor()),
        nodes=nodes
    )

    @timing
    def test_iter():
        return next(iter(iter_dataset))

    for i in range(10):
        test_iter()

    print_timing()
