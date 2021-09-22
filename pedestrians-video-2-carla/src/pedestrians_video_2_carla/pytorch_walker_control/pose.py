import time
from typing import Any, Dict, OrderedDict, Tuple

import carla
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

    def __pose_to_tensors(self, pose: OrderedDict[str, carla.Transform]) -> Tuple[Tensor, Tensor]:
        """
        Converts pose from OrderedDict to Tensor.

        :param pose: Pose as a { bone_name: carla.Transform} dict
        :type pose: OrderedDict
        :return: Pose points mapped to tuple of Tensors (locations, rotations).
            Angles are in radians and follow (roll, pitch, yaw) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
            In general this uses PyTorch3D coordinates system (right-handed, negative Z
            and angles when compared to CARLA).
        :rtype: Tuple[Tensor, Tensor]
        """

        locations, rotations = zip(*[(
            (p.location.x, p.location.y, -p.location.z),
            (np.deg2rad(-p.rotation.roll), np.deg2rad(
                -p.rotation.pitch), np.deg2rad(-p.rotation.yaw))
        ) for p in pose.values()])

        return (torch.tensor(locations, device=self._device, dtype=torch.float32),
                euler_angles_to_matrix(torch.tensor(rotations, device=self._device, dtype=torch.float32), "XYZ"))

    def __tensors_to_pose(self, p3d_locations: Tensor, p3d_rotations: Tensor) -> OrderedDict:
        pose = self.empty

        loc_list = list(
            p3d_locations.cpu().numpy().astype(float))
        rot_list = list(
            -np.rad2deg(matrix_to_euler_angles(p3d_rotations, "XYZ").cpu().numpy()).astype(float))

        for (idx, bone_name) in enumerate(pose.keys()):
            pose[bone_name] = carla.Transform(
                location=carla.Location(
                    x=loc_list[idx][0],
                    y=loc_list[idx][1],
                    z=-loc_list[idx][2]
                ),
                rotation=carla.Rotation(
                    pitch=rot_list[idx][1],
                    yaw=rot_list[idx][2],
                    roll=rot_list[idx][0]
                )
            )

        return pose

    def __move_to_relative(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Converts relative pose change (Rotations-only) to relative pose (Locations + Rotations).

        This method uses PyTorch3D coordinates system (right-handed, negative Z
            and radians (-roll, -pitch, -yaw) angles when compared to CARLA).

        :param x: (N, 3) tensor with the bone rotation changes.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be in radians and follow (roll, pitch, yaw) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
        :type x: Tensor
        :return: ((N, 3), (N, 3)) tensors with the relative bone transforms (locations, rotations).
        :rtype: Tuple[Tensor, Tensor]
        """

        # TODO: fix this

        reference_matrix = self.__relative_p3d_rotations
        change_matrix = euler_angles_to_matrix(x, "XYZ")

        self.__relative_p3d_rotations = torch.matmul(change_matrix, reference_matrix)
        return (self.__relative_p3d_locations, self.__relative_p3d_rotations)

    def __transform_descendants(self,
                                absolute_loc: Tensor,
                                absolute_rot: Tensor,
                                relative_loc: Tensor,
                                relative_rot: Tensor,
                                substructure: Dict[str, Any],
                                prev_loc: Tensor,
                                prev_rot: Tensor
                                ):
        # we shouldn't have more than one item here
        (bone_name, subsubstructures) = list(substructure.items())[0]
        idx = list(self.empty.keys()).index(bone_name)

        absolute_loc[idx] = Rotate(prev_rot).compose(
            Translate(prev_loc.unsqueeze(0))).transform_points(relative_loc[idx].unsqueeze(0))
        absolute_rot[idx] = torch.matmul(relative_rot[idx], prev_rot)

        if subsubstructures is not None:
            for subsubstructure in subsubstructures:
                self.__transform_descendants(
                    absolute_loc,
                    absolute_rot,
                    relative_loc,
                    relative_rot,
                    subsubstructure,
                    absolute_loc[idx],
                    absolute_rot[idx]
                )

    def __relative_to_absolute(self, loc: Tensor, rot: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Converts relative pose (Locations + Rotations) to absolute pose (Locations + Rotations).

        This method uses PyTorch3D coordinates system (right-handed, negative Z
            and radians (-roll, -pitch, -yaw) angles when compared to CARLA).

        :param loc: (N, 3) tensor with the relative bone locations (x, y, -z).
            Order of bones needs to follow the order of keys as returned by Pose.empty.
        :type loc: Tensor
        :param rot: (N, 3) tensor with the relative bone rotations.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be in radians and follow (roll, pitch, yaw) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
        :type rot: Tensor
        :return: ((N, 3), (N, 3)) tensors with the bone locations and rotations (in radians)
            relative to the root point ((x, y, -z), (-roll, -pitch, -yaw)).
        :rtype: Tuple[Tensor, Tensor]
        """
        absolute_loc = torch.zeros_like(loc)
        absolute_rot = torch.zeros_like(rot)

        # do it recursively for now
        self.__transform_descendants(
            absolute_loc,
            absolute_rot,
            loc,
            rot,
            self._structure[0],
            torch.tensor((0, 0, 0), device=loc.device),
            torch.eye(3, device=rot.device)
        )

        return absolute_loc, absolute_rot

    def forward(self, x: Tensor) -> Tensor:
        """
        Converts relative pose change (Rotations-only) to absolute pose (Locations-only).

        This method uses PyTorch3D coordinates system (right-handed, negative Z
            and radians (-roll, -pitch, -yaw) angles when compared to CARLA).

        :param x: (N, 3) tensor with the bone rotation changes.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be in radians and follow (roll, pitch, yaw) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!

        :type x: Tensor
        :return: (N, 3) tensor with the bone locations relative to the root point (x, y, -z).
        :rtype: Tensor
        """
        loc, rot = self.__move_to_relative(x.to(self._device))
        a_lot, _ = self.__relative_to_absolute(loc, rot)
        return a_lot

    @property
    def relative(self):
        if (self._last_rel_mod != self.__last_rel_get):
            self.__last_rel = self.__tensors_to_pose(
                self.__relative_p3d_locations, self.__relative_p3d_rotations)
            self.__last_rel_get = self._last_rel_mod
        return self.__last_rel

    @relative.setter
    def relative(self, new_pose_dict):
        # update our reference Tensors
        (self.__relative_p3d_locations,
         self.__relative_p3d_rotations) = self.__pose_to_tensors(new_pose_dict)
        # and set the timestamp
        self._last_rel_mod = time.time()

    @property
    def absolute(self):
        if self._last_abs_mod != self._last_rel_mod:
            absolute_loc, absolute_rot = self.__relative_to_absolute(
                self.__relative_p3d_locations, self.__relative_p3d_rotations)
            absolute_pose = self.__tensors_to_pose(absolute_loc, absolute_rot)

            self._last_abs = absolute_pose
            self._last_abs_mod = self._last_rel_mod

        return self._deepcopy_pose_dict(self._last_abs)
