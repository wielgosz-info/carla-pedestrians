from typing import Union

import carla
from scipy.spatial.transform import Rotation, rotation


def carla_to_scipy_rotation(rotation: Union[carla.Rotation, carla.Transform]) -> Rotation:
    """
    Converts carla.Rotation or carla.Transform to scipy.spatial.transform.Rotation
    """

    if isinstance(rotation, carla.Transform):
        rotation = rotation.rotation

    # we need to follow UE4 order of axes
    return Rotation.from_euler('ZYX', [
        rotation.yaw,
        rotation.pitch,
        rotation.roll,
    ], degrees=True)


def scipy_to_carla_rotation(rotation: Rotation) -> carla.Rotation:
    """
    Converts scipy.spatial.transform.Rotation to carla.Rotation
    """

    (yaw, pitch, roll) = rotation.as_euler('ZYX', degrees=True)
    return carla.Rotation(
        pitch=pitch,
        yaw=yaw,
        roll=roll
    )


def mul_carla_rotations(reference_rotation: carla.Rotation, local_rotation: carla.Rotation) -> carla.Rotation:
    """
    Multiplies two carla.Rotation objects by converting them to scipy.spatial.transform.Rotation
    and then converting the result back (since carla.Rotation API doesn't offer that option).
    """

    reference_rot = carla_to_scipy_rotation(reference_rotation)
    local_rot = carla_to_scipy_rotation(local_rotation)

    # and now multiply & convert it back
    return scipy_to_carla_rotation(reference_rot*local_rot)


def deepcopy_location(v: carla.Location) -> carla.Location:
    return carla.Location(
        x=v.x,
        y=v.y,
        z=v.z
    )


def deepcopy_rotation(v: carla.Rotation) -> carla.Rotation:
    return carla.Rotation(
        pitch=v.pitch,
        yaw=v.yaw,
        roll=v.roll,
    )


def deepcopy_transform(v: carla.Transform) -> carla.Transform:
    return carla.Transform(
        location=deepcopy_location(v.location),
        rotation=deepcopy_rotation(v.rotation)
    )
