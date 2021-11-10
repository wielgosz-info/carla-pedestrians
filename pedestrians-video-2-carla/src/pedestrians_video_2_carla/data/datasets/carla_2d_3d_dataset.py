from typing import Callable
import torch
from torch.utils.data import IterableDataset, Dataset
import h5pickle as h5py
from pedestrians_video_2_carla.modules.projection.projection import ProjectionModule
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
import numpy as np
from torch.functional import Tensor
from pytorch3d.transforms import euler_angles_to_matrix


class Carla2D3DDataset(Dataset):
    def __init__(self, set_filepath: str, nodes: CARLA_SKELETON = CARLA_SKELETON, transform=None, **kwargs) -> None:
        set_file = h5py.File(set_filepath, 'r')

        self.projection_2d = set_file['carla_2d_3d/projection_2d']
        self.pose_changes = set_file['carla_2d_3d/targets/pose_changes']
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
                'absolute_pose_loc': absolute_pose_loc,
                'absolute_pose_rot': absolute_pose_rot,
            },
            meta
        )


class Carla2D3DIterableDataset(IterableDataset):
    def __init__(self,
                 batch_size: int = 64,
                 clip_length: int = 30,
                 random_changes_each_frame=3,
                 max_change_in_deg=5,
                 nodes: CARLA_SKELETON = CARLA_SKELETON,
                 transform: Callable[[Tensor], Tensor] = None,
                 **kwargs) -> None:
        self.transform = transform
        self.nodes = nodes
        self.clip_length = clip_length
        self.random_changes_each_frame = random_changes_each_frame
        self.max_change_in_rad = np.deg2rad(max_change_in_deg)
        self.batch_size = batch_size

        self.projection = ProjectionModule(
            input_nodes=self.nodes,
            output_nodes=self.nodes,
            projection_transform=self.transform
        )

    def __iter__(self):
        # this is infinite generative dataset, it doesn't matter how many workers are there
        while True:
            inputs, targets, meta = self.__generate_batch()
            for idx in range(self.batch_size):
                yield (
                    inputs[idx],
                    {k: v[idx] for k, v in targets.items()},
                    {k: v[idx] for k, v in meta.items()}
                )

    def __generate_batch(self):
        nodes_size = len(self.nodes)
        nodes_nums = np.arange(nodes_size)
        pose_changes = torch.zeros(
            (self.batch_size, self.clip_length, nodes_size, 3))

        for idx in range(self.batch_size):
            for i in range(self.clip_length):
                indices = np.random.choice(nodes_nums,
                                           size=self.random_changes_each_frame, replace=False)
                pose_changes[idx, i, indices] = (torch.rand(
                    (self.random_changes_each_frame, 3)) * 2 - 1) * self.max_change_in_rad
        pose_changes_matrix = euler_angles_to_matrix(pose_changes, "XYZ")

        # TODO: we should probably take care of the "correct" pedestrians data distribution
        # need to find some pedestrian statistics
        age = np.random.choice(['adult', 'child'], size=self.batch_size)
        gender = np.random.choice(['male', 'female'], size=self.batch_size)

        self.projection.on_batch_start((pose_changes_matrix, None, {
            'age': age,
            'gender': gender
        }), 0, None)
        projection_2d, absolute_pose_loc, absolute_pose_rot = self.projection.project_pose(
            pose_changes_matrix
        )

        # use the third dimension as 'confidence' of the projection
        # so we're compatible with OpenPose
        # this will also prevent the models from accidentally using
        # the depth data that pytorch3d leaves in the projections
        projection_2d[..., 2] = 1.0

        if self.transform:
            projection_2d = self.transform(projection_2d)

        return (
            projection_2d,
            {
                'pose_changes': pose_changes_matrix,
                'absolute_pose_loc': absolute_pose_loc,
                'absolute_pose_rot': absolute_pose_rot
            },
            {'age': age, 'gender': gender}
        )


if __name__ == "__main__":
    from pedestrians_video_2_carla.utils.timing import timing, print_timing
    from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckNormalize, CarlaHipsNeckExtractor

    nodes = CARLA_SKELETON
    iter_dataset = Carla2D3DIterableDataset(
        batch_size=256,
        clip_length=180,
        transform=HipsNeckNormalize(CarlaHipsNeckExtractor(nodes)),
        nodes=nodes
    )

    @timing
    def test_iter():
        return next(iter(iter_dataset))

    for i in range(10):
        test_iter()

    print_timing()
