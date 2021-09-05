import pytest

import numpy as np

from pedestrians_video_2_carla.walker_control.io import load_reference
from pedestrians_video_2_carla.walker_control.transforms import unreal_to_carla
from pedestrians_video_2_carla.walker_control.pose import Pose


def test_set_pose():
    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    relative_pose = unreal_to_carla(unreal_rel_pose['transforms'])

    p = Pose()
    p.relative = relative_pose

    for bone_name, transforms_dict in relative_pose.items():
        assert np.isclose(p.relative[bone_name].location.x, transforms_dict.location.x)
        assert np.isclose(p.relative[bone_name].location.y, transforms_dict.location.y)
        assert np.isclose(p.relative[bone_name].location.z, transforms_dict.location.z)
        assert np.isclose(p.relative[bone_name].rotation.pitch,
                          transforms_dict.rotation.pitch)
        assert np.isclose(p.relative[bone_name].rotation.yaw,
                          transforms_dict.rotation.yaw)
        assert np.isclose(p.relative[bone_name].rotation.roll,
                          transforms_dict.rotation.roll)


def test_relative_to_absolute():
    unreal_abs_pose = load_reference('sk_female_absolute.yaml')
    absolute_pose = unreal_to_carla(unreal_abs_pose['transforms'])

    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    relative_pose = unreal_to_carla(unreal_rel_pose['transforms'])

    p = Pose()
    p.relative = relative_pose

    for bone_name, transforms_dict in absolute_pose.items():
        assert np.isclose(p.absolute[bone_name].location.x,
                          transforms_dict.location.x, atol=1e-5)
        assert np.isclose(p.absolute[bone_name].location.y,
                          transforms_dict.location.y, atol=1e-5)
        assert np.isclose(p.absolute[bone_name].location.z,
                          transforms_dict.location.z, atol=1e-5)
        assert np.isclose(p.absolute[bone_name].rotation.pitch,
                          transforms_dict.rotation.pitch, atol=1e-2)
        assert np.isclose(p.absolute[bone_name].rotation.yaw,
                          transforms_dict.rotation.yaw, atol=1e-2)
        assert np.isclose(p.absolute[bone_name].rotation.roll,
                          transforms_dict.rotation.roll, atol=1e-2)
