from typing import Union
import numpy as np
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from pedestrians_video_2_carla.walker_control.pose import Pose


def test_get_relative_pose(relative_pose, reference_pose: Union[Pose, P3dPose]):
    relative = reference_pose.relative

    for bone_name, transforms_dict in relative_pose.items():
        assert np.isclose(relative[bone_name].location.x, transforms_dict.location.x)
        assert np.isclose(relative[bone_name].location.y, transforms_dict.location.y)
        assert np.isclose(relative[bone_name].location.z, transforms_dict.location.z)
        assert np.isclose(relative[bone_name].rotation.pitch,
                          transforms_dict.rotation.pitch)
        assert np.isclose(relative[bone_name].rotation.yaw,
                          transforms_dict.rotation.yaw)
        assert np.isclose(relative[bone_name].rotation.roll,
                          transforms_dict.rotation.roll)


def test_relative_to_absolute(absolute_pose, reference_pose: Union[Pose, P3dPose]):
    absolute = reference_pose.absolute

    for bone_name, transforms_dict in absolute_pose.items():
        assert np.isclose(absolute[bone_name].location.x,
                          transforms_dict.location.x, atol=1e-5)
        assert np.isclose(absolute[bone_name].location.y,
                          transforms_dict.location.y, atol=1e-5)
        assert np.isclose(absolute[bone_name].location.z,
                          transforms_dict.location.z, atol=1e-5)
        assert np.isclose(absolute[bone_name].rotation.pitch,
                          transforms_dict.rotation.pitch, atol=1e-2)
        assert np.isclose(absolute[bone_name].rotation.yaw,
                          transforms_dict.rotation.yaw, atol=1e-2)
        assert np.isclose(absolute[bone_name].rotation.roll,
                          transforms_dict.rotation.roll, atol=1e-2)
