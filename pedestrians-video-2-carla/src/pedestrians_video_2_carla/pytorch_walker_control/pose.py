from typing import Any, Dict, OrderedDict, Tuple
import carla
from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles
from scipy.spatial import transform
import torch
from pytorch3d.transforms import (euler_angles_to_matrix, matrix_to_quaternion)
import numpy as np
from torch.types import Device
from pedestrians_video_2_carla.walker_control.pose import Pose
from torch.functional import Tensor
from pytorch3d.transforms.transform3d import Transform3d


class P3dPose(Pose, torch.nn.Module):
    def __init__(self, device: Device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._device = device
        self.__relative_p3d_locations = None
        self.__relative_p3d_rotations = None

    def __relative_to_tensors(self, relative: OrderedDict[str, carla.Transform] = None) -> Tuple[Tensor, Tensor]:
        """
        Converts pose from OrderedDict to Tensor.

        :param relative: Relative pose as a { bone_name: carla.Transform} dict,
            when None the current **relative** pose will be used; defaults to None
        :type relative: OrderedDict, optional
        :return: Pose points mapped to tuple of Tensors (locations, rotations).
            Angles are in radians and follow (yaw, pitch, roll) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
        :rtype: Tuple[Tensor, Tensor]
        """

        if relative is None:
            relative = self.relative

        locations, rotations = zip(*[(
            (p.location.x, p.location.y, p.location.z),
            (p.rotation.yaw, p.rotation.pitch, p.rotation.roll)
        ) for p in relative.values()])

        return (torch.tensor(locations, device=self._device),
                torch.tensor(rotations, device=self._device) * np.pi / 180.)

    def __move_to_relative(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Converts relative pose change (Rotations-only) to relative pose (Locations + Rotations).

        :param x: (N, 3) tensor with the bone rotation changes.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be in radians and follow (yaw, pitch, roll) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
        :type x: Tensor
        :return: ((N, 3), (N, 3)) tensors with the relative bone transforms (locations, rotations).
        :rtype: Tuple[Tensor, Tensor]
        """

        reference_matrix = euler_angles_to_matrix(self.__relative_p3d_rotations, "ZYX")
        change_matrix = euler_angles_to_matrix(x, "ZYX")
        return (self.__relative_p3d_locations,
                matrix_to_euler_angles(reference_matrix*change_matrix, "ZYX"))

    def __transform_descendants(self,
                                absolute_loc: Tensor,
                                absolute_rot: Tensor,
                                relative_loc: Tensor,
                                relative_rot: Tensor,
                                substructure: Dict[str, Any],
                                prev_transform: Transform3d,
                                prev_loc: Tensor,
                                prev_rot_matrix: Tensor
                                ):
        # we shouldn't have more than one item here
        (bone_name, subsubstructures) = list(substructure.items())[0]
        idx = list(self.empty.keys()).index(bone_name)

        relative_rot_matrix = euler_angles_to_matrix(relative_rot[idx], "ZYX")

        # shift = prev_transform.transform_points(relative_loc[idx].unsqueeze(0))

        # absolute_loc[idx] = prev_loc + shift

        # TODO: does CARLA actually need this info?
        # absolute_rot[idx] = torch.matmul(prev_rot_matrix, relative_rot_matrix)

        new_transform = prev_transform.translate(
            relative_loc[idx].unsqueeze(0)).rotate(relative_rot_matrix)
        absolute_loc[idx] = new_transform.transform_points(torch.tensor(
            [[0, 0, 0]], dtype=absolute_loc.dtype, device=absolute_loc.device))

        if subsubstructures is not None:
            for subsubstructure in subsubstructures:
                self.__transform_descendants(
                    absolute_loc,
                    absolute_rot,
                    relative_loc,
                    relative_rot,
                    subsubstructure,
                    new_transform,
                    absolute_loc[idx],
                    absolute_rot[idx]
                )

    def __relative_to_absolute(self, loc: Tensor, rot: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Converts relative pose (Locations + Rotations) to absolute pose (Locations + Rotations).

        :param loc: (N, 3) tensor with the relative bone locations (x, y, z).
            Order of bones needs to follow the order of keys as returned by Pose.empty.
        :type loc: Tensor
        :param rot: (N, 3) tensor with the relative bone rotations.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be in radians and follow (yaw, pitch, roll) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
        :type rot: Tensor
        :return: ((N, 3), (N, 3)) tensors with the bone locations and rotations (in radians)
            relative to the root point ((x, y, z), (yaw, pitch, roll)).
        :rtype: Tuple[Tensor, Tensor]
        """
        absolute_loc = torch.zeros_like(loc)

        # Do we need this?
        absolute_rot_matrix = torch.zeros(
            size=(*rot.shape[:-1], 3, 3), device=rot.device)

        # do it recursively for now
        self.__transform_descendants(
            absolute_loc,
            absolute_rot_matrix,
            loc,
            rot,
            self._structure[0],
            Transform3d(dtype=loc.dtype, device=loc.device),
            torch.tensor((0, 0, 0), device=loc.device),
            euler_angles_to_matrix(torch.tensor((0, 0, 0), device=rot.device), "ZYX")
        )

        return absolute_loc, matrix_to_euler_angles(absolute_rot_matrix, "ZYX")

    def forward(self, x: Tensor) -> Tensor:
        """
        Converts relative pose change (Rotations-only) to absolute pose (Locations-only).

        :param x: (N, 3) tensor with the bone rotation changes.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be in radians and follow (yaw, pitch, roll) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
        :type x: Tensor
        :return: (N, 3) tensor with the bone locations relative to the root point (x, y, z).
        :rtype: Tensor
        """
        loc, rot = self.__move_to_relative(x.to(self._device))
        a_lot, _ = self.__relative_to_absolute(loc, rot)
        return a_lot

    @Pose.relative.setter
    def relative(self, new_pose_dict):
        # the following calls ase Pose class setter
        Pose.relative.fset(self, new_pose_dict)
        # and then we update our reference Tensors
        (self.__relative_p3d_locations,
         self.__relative_p3d_rotations) = self.__relative_to_tensors()

    @property
    def absolute(self):
        if self._last_abs_mod != self._last_rel_mod:
            # ensure bones in absolute pose will be in the same order as they were in relative
            # this will be updated in-place
            absolute_pose = self.empty

            absolute_loc, absolute_rot = self.__relative_to_absolute(
                self.__relative_p3d_locations, self.__relative_p3d_rotations)
            absolute_loc_list = list(absolute_loc.cpu().numpy().astype(float))
            absolute_rot_list = list(
                (absolute_rot.cpu().numpy() * 180. / np.pi).astype(float))

            for (idx, bone_name) in enumerate(absolute_pose.keys()):
                absolute_pose[bone_name] = carla.Transform(
                    location=carla.Location(*absolute_loc_list[idx]),
                    rotation=carla.Rotation(
                        pitch=absolute_rot_list[idx][1],
                        yaw=absolute_rot_list[idx][0],
                        roll=absolute_rot_list[idx][2]
                    )
                )

            self._last_abs = absolute_pose
            self._last_abs_mod = self._last_rel_mod

        return self._deepcopy_pose_dict(self._last_abs)
