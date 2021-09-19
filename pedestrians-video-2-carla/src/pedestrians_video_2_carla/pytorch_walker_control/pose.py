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
        (self.__relative_p3d_locations,
         self.__relative_p3d_rotations) = self.__relative_to_tensors()

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

        return (torch.Tensor(locations, device=self._device),
                torch.Tensor(rotations, device=self._device) * np.pi / 180.)

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
                                relative_loc: Tensor,
                                relative_rot: Tensor,
                                substructure: Dict[str, Any],
                                prev_transform: Transform3d
                                ):
        # we shouldn't have more than one item here
        (idx, subsubstructures) = list(enumerate(substructure.values()[0]))

        absolute_loc[idx] = prev_transform.transform_points(relative_loc[idx])
        new_transform = prev_transform.translate(relative_loc[idx]).rotate(
            euler_angles_to_matrix(relative_rot[idx], "ZYX"))

        if subsubstructures is not None:
            for subsubstructure in subsubstructures:
                self.__transform_descendants(
                    absolute_loc,
                    relative_loc,
                    relative_rot,
                    subsubstructure,
                    new_transform
                )

    def __relative_to_absolute(self, loc: Tensor, rot: Tensor) -> Tensor:
        """
        Converts relative pose (Locations + Rotations) to absolute pose (Locations-only).

        :param loc: (N, 3) tensor with the relative bone locations (x, y, z).
            Order of bones needs to follow the order of keys as returned by Pose.empty.
        :type loc: Tensor
        :param rot: (N, 3) tensor with the relative bone rotations (yaw, pitch, roll).
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be specified in radians!
        :type rot: Tensor
        :return:  (N, 3) tensor with the bone locations relative to the root point (x, y, z).
        :rtype: Tensor
        """
        absolute_loc = torch.zeros_like(loc)

        # do it recursively for now
        self.__transform_descendants(
            absolute_loc,
            loc,
            rot,
            self._structure[0],
            Transform3d(dtype=loc.dtype, device=loc.device)
        )

        return absolute_loc

    def forward(self, x: Tensor) -> Tensor:
        """
        Converts relative pose change (Rotations-only) to absolute pose (Locations-only).

        :param x: (N, 3) tensor with the bone rotation changes (pitch, yaw, roll).
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be specified in radians!
        :type x: Tensor
        :return: (N, 3) tensor with the bone locations relative to the root point (x, y, z).
        :rtype: Tensor
        """
        loc, rot = self.__move_to_relative(x.to(self._device))
        return self.__relative_to_absolute(loc, rot)
