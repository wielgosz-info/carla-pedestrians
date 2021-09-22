from typing import Any, Dict, OrderedDict, Tuple
import carla
from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles, quaternion_multiply, quaternion_to_matrix
import torch
from pytorch3d.transforms import (euler_angles_to_matrix, matrix_to_quaternion)
import numpy as np
from torch.types import Device
from pedestrians_video_2_carla.utils.spatial import mul_rotations
from pedestrians_video_2_carla.walker_control.pose import Pose
from torch.functional import Tensor
from pytorch3d.transforms.transform3d import Rotate, Transform3d, Translate


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
            Angles are in radians and follow (roll, pitch, yaw) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
        :rtype: Tuple[Tensor, Tensor]
        """

        if relative is None:
            relative = self.relative

        locations, rotations = zip(*[(
            (p.location.x, p.location.y, -p.location.z),
            (np.deg2rad(-p.rotation.roll), np.deg2rad(
                -p.rotation.pitch), np.deg2rad(-p.rotation.yaw))
        ) for p in relative.values()])

        return (torch.tensor(locations, device=self._device, dtype=torch.float32),
                euler_angles_to_matrix(torch.tensor(rotations, device=self._device, dtype=torch.float32), "XYZ"))

    def __move_to_relative(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Converts relative pose change (Rotations-only) to relative pose (Locations + Rotations).

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
        return (self.__relative_p3d_locations,
                matrix_to_euler_angles(reference_matrix*change_matrix, "XYZ"))

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

        :param loc: (N, 3) tensor with the relative bone locations (x, y, z).
            Order of bones needs to follow the order of keys as returned by Pose.empty.
        :type loc: Tensor
        :param rot: (N, 3) tensor with the relative bone rotations.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be in radians and follow (roll, pitch, yaw) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
        :type rot: Tensor
        :return: ((N, 3), (N, 3)) tensors with the bone locations and rotations (in radians)
            relative to the root point ((x, y, z), (roll, pitch, yaw)).
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

        :param x: (N, 3) tensor with the bone rotation changes.
            Order of bones needs to follow the order of keys as returned by Pose.empty.
            Angles should be in radians and follow (roll, pitch, yaw) order as opposed to
            CARLA degrees and (pitch, yaw, roll) order!
        :type x: Tensor
        :return: (N, 3) tensor with the bone locations relative to the root point (x, y, z).
        :rtype: Tensor
        """
        loc, rot = self.__move_to_relative(x.to(self._device))
        a_lot, _ = self.__relative_to_absolute(loc, rot)
        return a_lot

    @ Pose.relative.setter
    def relative(self, new_pose_dict):
        # the following calls ase Pose class setter
        Pose.relative.fset(self, new_pose_dict)
        # and then we update our reference Tensors
        (self.__relative_p3d_locations,
         self.__relative_p3d_rotations) = self.__relative_to_tensors()

    @ property
    def absolute(self):
        if self._last_abs_mod != self._last_rel_mod:
            # ensure bones in absolute pose will be in the same order as they were in relative
            # this will be updated in-place
            absolute_pose = self.empty

            absolute_loc, absolute_rot = self.__relative_to_absolute(
                self.__relative_p3d_locations, self.__relative_p3d_rotations)

            absolute_loc_list = list(absolute_loc.cpu().numpy().astype(float))
            absolute_rot_list = list(
                -np.rad2deg(matrix_to_euler_angles(absolute_rot, "XYZ").cpu().numpy()).astype(float))

            for (idx, bone_name) in enumerate(absolute_pose.keys()):
                absolute_pose[bone_name] = carla.Transform(
                    location=carla.Location(
                        x=absolute_loc_list[idx][0],
                        y=absolute_loc_list[idx][1],
                        z=-absolute_loc_list[idx][2]
                    ),
                    rotation=carla.Rotation(
                        pitch=absolute_rot_list[idx][1],
                        yaw=absolute_rot_list[idx][2],
                        roll=absolute_rot_list[idx][0]
                    )
                )

            self._last_abs = absolute_pose
            self._last_abs_mod = self._last_rel_mod

        return self._deepcopy_pose_dict(self._last_abs)
