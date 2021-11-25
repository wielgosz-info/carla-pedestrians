import time
from typing import Any, Dict, OrderedDict, Tuple

import warnings

try:
    import carla
except ImportError:
    import pedestrians_video_2_carla.carla_utils.mock_carla as carla
    warnings.warn("Using mock carla.", ImportWarning)


import numpy as np
import torch
from torch.functional import Tensor
from pedestrians_video_2_carla.walker_control.pose import Pose
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles
from pytorch3d.transforms.transform3d import Rotate, Translate
from torch.types import Device


class P3dPose(Pose, torch.nn.Module):
    def __init__(self, device: Device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._device = device
        self.__relative_p3d_locations = None
        self.__relative_p3d_rotations = None

        self.__last_rel = None
        self.__last_rel_get = None

    def pose_to_tensors(self, pose: OrderedDict[str, carla.Transform]) -> Tuple[Tensor, Tensor]:
        """
        Converts pose from OrderedDict to Tensor.

        :param pose: Pose as a { bone_name: carla.Transform} dict
        :type pose: OrderedDict
        :return: Pose points mapped to tuple of Tensors (locations, rotations).
            Rotations are specified as rotation matrix.
            In general this uses PyTorch3D coordinates system (right-handed, negative Z
            and negative angles when compared to CARLA).
        :rtype: Tuple[Tensor, Tensor]
        """

        locations, rotations = zip(*[(
            (p.location.x, p.location.y, -p.location.z),
            (np.deg2rad(-p.rotation.roll), np.deg2rad(
                -p.rotation.pitch), np.deg2rad(-p.rotation.yaw))
        ) for p in pose.values()])

        return (torch.tensor(locations, device=self._device, dtype=torch.float32),
                euler_angles_to_matrix(torch.tensor(rotations, device=self._device, dtype=torch.float32), "XYZ"))

    def tensors_to_pose(self, p3d_locations: Tensor, p3d_rotations: Tensor) -> OrderedDict:
        """
        Converts pose from Tensor to OrderedDict.

        Tensors can be either absolute or relative. If relative, the resulting pose also is relative.

        :param p3d_locations: (B, 3) tensor with the bone locations.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            In general this uses PyTorch3D coordinates system (right-handed, negative Z
            and negative angles when compared to CARLA).
        :type p3d_locations: Tensor
        :param p3d_rotations: (B, 3, 3) tensor with the bone rotations as rotation matrices.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
        :type p3d_rotations: Tensor
        :return: Pose as a { bone_name: carla.Transform } OrderedDict
        """
        pose = self.empty

        loc_list = list(
            p3d_locations.cpu().numpy().astype(float))

        rot_list = list(
            -np.rad2deg(matrix_to_euler_angles(p3d_rotations, "XYZ").cpu().numpy()).astype(float))

        for (idx, bone_name) in enumerate(pose.keys()):
            loc = carla.Location(
                x=loc_list[idx][0],
                y=loc_list[idx][1],
                z=-loc_list[idx][2]
            )
            rot = carla.Rotation(
                pitch=rot_list[idx][1],
                yaw=rot_list[idx][2],
                roll=rot_list[idx][0]
            )
            pose[bone_name] = carla.Transform(
                location=loc,
                rotation=rot
            )

        return pose

    def __move_to_relative(self, changes_matrix: Tensor, prev_relative_rotations: Tensor) -> Tensor:
        """
        Converts relative pose change (Rotations-only) to relative pose Rotations.

        :param changes: (N, B, 3, 3) tensor with the bone rotation changes as rotation matrices.
        :type changes: Tensor
        :param prev_relative_rotations: (N, B, 3, 3) tensor with the previous bone rotations as rotation matrices.
        :type prev_relative_rotations: Tensor
        :return: (N, B, 3, 3) tensors with the relative bone rotations as rotation matrices.
        :rtype: Tensor
        """

        bs = changes_matrix.shape[0]
        return torch.bmm(
            changes_matrix.reshape((-1, 3, 3)),
            prev_relative_rotations.reshape((-1, 3, 3))
        ).reshape((bs, -1, 3, 3))

    def __transform_descendants(self,
                                absolute_loc: Tensor,
                                absolute_rot: Tensor,
                                relative_loc: Tensor,
                                relative_rot: Tensor,
                                substructure: Dict[str, Any],
                                prev_transform: Tensor
                                ):
        # we shouldn't have more than one item here
        (bone_name, subsubstructures) = list(substructure.items())[0]
        idx = list(self.empty.keys()).index(bone_name)

        pad_rel_loc = torch.nn.functional.pad(
            relative_loc[:, idx:idx+1], pad=(0, 1, 0, 0), mode='constant', value=1)
        abs_loc = torch.bmm(pad_rel_loc, prev_transform)
        absolute_loc[:, idx] = abs_loc[:, 0, :3]
        absolute_rot[:, idx] = torch.bmm(
            relative_rot[:, idx], prev_transform[:, :3, :3])

        new_transform = torch.eye(4, device=self._device).reshape(
            (1, 4, 4)).repeat((absolute_loc.shape[0], 1, 1))
        new_transform[:, :3, :3] = absolute_rot[:, idx]
        new_transform[:, 3, :3] = absolute_loc[:, idx]

        if subsubstructures is not None:
            for subsubstructure in subsubstructures:
                self.__transform_descendants(
                    absolute_loc,
                    absolute_rot,
                    relative_loc,
                    relative_rot,
                    subsubstructure,
                    new_transform
                )

    def __relative_to_absolute(self, loc: Tensor, rot: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Converts relative pose (Locations + Rotations) to absolute pose (Locations + Rotations).

        This method uses PyTorch3D coordinates system (right-handed, negative Z
            and radians (-roll, -pitch, -yaw) angles when compared to CARLA).

        :param loc: (N, B, 3) tensor with the relative bone locations (x, y, -z).
            Order of bones needs to follow the order of keys as returned by Pose.empty.
        :type loc: Tensor
        :param rot: (N, B, 3, 3) tensor with the relative bone rotation matrices.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
        :type rot: Tensor
        :return: ((N, B, 3), (N, B, 3, 3)) tensors with the bone locations and rotation matrices
            relative to the root point ((x, y, -z), (-roll, -pitch, -yaw)).
        :rtype: Tuple[Tensor, Tensor]
        """
        absolute_loc = torch.zeros_like(loc)
        absolute_rot = torch.zeros_like(rot)

        initial_transform = torch.eye(4, device=self._device).reshape(
            (1, 4, 4)).repeat((absolute_loc.shape[0], 1, 1))

        # do it recursively for now
        self.__transform_descendants(
            absolute_loc,
            absolute_rot,
            loc,
            rot,
            self._structure[0],
            initial_transform
        )

        return absolute_loc, absolute_rot

    def forward(self, changes_matrix: Tensor, prev_relative_loc: Tensor, prev_relative_rot: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Converts relative pose change (Rotations-only) to absolute pose (Locations-only).

        :param changes_matrix: (N, B, 3, 3) tensor with the bone rotation changes as rotation matrices.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
        :type changes_matrix: Tensor
        :param prev_relative_loc: (N, B, 3) tensor with the relative bone locations (x, y, -z).
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            This normally is only initialized once per pedestrian and never changes.
            Get it by calling `P3dPose.tensors`.
        :type prev_relative_loc: Tensor
        :param prev_relative_rot: (N, B, 3, 3) tensor with the relative bone rotation matrices.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            This normally is initialized when the pedestrian is created and **changes after each movement**.
            Get it by calling `P3dPose.tensors`.
        :type prev_relative_rot: Tensor

        :return: ((N, B, 3), (N, B, 3, 3), (N, B, 3, 3)) tuple of tensors containing updated data:
            - absolute bone locations (relative to the root point),
            - absolute bone rotations (relative to the root point),
            - updated relative bone rotations (which is what you will feed into this method for the next movement),
            When converting to CARLA location coords are in (x, y, -z) order.
        :rtype: Tuple[Tensor, Tensor, Tensor]
        """
        rot = self.__move_to_relative(changes_matrix, prev_relative_rot)
        a_lot, a_rot = self.__relative_to_absolute(prev_relative_loc, rot)
        return a_lot, a_rot, rot

    @property
    def relative(self):
        if (self._last_rel_mod != self.__last_rel_get):
            self.__last_rel = self.tensors_to_pose(
                self.__relative_p3d_locations, self.__relative_p3d_rotations)
            self.__last_rel_get = self._last_rel_mod
        return self.__last_rel

    @relative.setter
    def relative(self, new_pose_dict):
        # update our reference Tensors
        (self.__relative_p3d_locations,
         self.__relative_p3d_rotations) = self.pose_to_tensors(new_pose_dict)
        # and set the timestamp
        self._last_rel_mod = time.time_ns()

    @property
    def absolute(self):
        if self._last_abs_mod != self._last_rel_mod:
            absolute_loc, absolute_rot = self.__relative_to_absolute(
                self.__relative_p3d_locations.unsqueeze(0), self.__relative_p3d_rotations.unsqueeze(0))
            absolute_pose = self.tensors_to_pose(absolute_loc[0], absolute_rot[0])

            self._last_abs = absolute_pose
            self._last_abs_mod = self._last_rel_mod

        return self._deepcopy_pose_dict(self._last_abs)

    def move(self, rotations: Dict[str, carla.Rotation]):
        # we need correct bones indexes
        bone_names = list(self.empty.keys())
        # and default no-change for each bone
        changes = torch.zeros((len(bone_names), 3),
                              device=self._device, dtype=torch.float32)

        # for each defined rotation, we merge it with the current one
        for bone_name, rotation_change in rotations.items():
            idx = bone_names.index(bone_name)
            changes[idx] = torch.tensor((
                np.deg2rad(-rotation_change.roll),
                np.deg2rad(-rotation_change.pitch),
                np.deg2rad(-rotation_change.yaw)
            ), device=self._device, dtype=torch.float32)

            changes_matrix = euler_angles_to_matrix(changes, "XYZ")

        self.__relative_p3d_rotations = self.__move_to_relative(
            changes_matrix.unsqueeze(0), self.__relative_p3d_rotations.unsqueeze(0))[0]
        self._last_rel_mod = time.time_ns()

    @property
    def tensors(self) -> Tuple[Tensor, Tensor]:
        """
        Get current (relative) position tensors. The tensors have gradients disabled.
        Rotations are returned in the form of rotation matrices.

        :return: (locations, rotations) clones of the current position tensors.
        :rtype: Tuple[Tensor, Tensor]
        """
        return (
            self.__relative_p3d_locations.detach().clone(),
            self.__relative_p3d_rotations.detach().clone(),
        )

    @tensors.setter
    def tensors(self, tensors: Tuple[Tensor, Tensor]):
        # update our reference Tensors
        (self.__relative_p3d_locations,
         self.__relative_p3d_rotations) = tensors
        # and set the timestamp
        self._last_rel_mod = time.time_ns()
