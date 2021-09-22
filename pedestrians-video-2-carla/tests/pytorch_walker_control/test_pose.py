"""
This file contains additional tests for pytorch3d specific implementation.
The common API tests are in tests.walker_control.test_pose.
"""

import numpy as np
import carla
import random
from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.walker_control.pose import Pose


def test_p3d_move_matches_base_move(relative_pose, device):
    """
    Tests movement calculations by comparing them with "base" Pose implementation
    """

    # set
    p3d_pose = P3dPose(device=device)
    p3d_pose.relative = relative_pose

    base_pose = Pose()
    base_pose.relative = relative_pose

    # apply some movements
    for _ in range(3):
        movement = {
            'crl_shoulder__L': carla.Rotation(yaw=random.random()*5-10, pitch=random.random()*5-10, roll=random.random()*5-10),
            'crl_arm__L': carla.Rotation(yaw=random.random()*5-10, pitch=random.random()*5-10, roll=random.random()*5-10),
            'crl_foreArm__L': carla.Rotation(yaw=random.random()*5-10, pitch=random.random()*5-10, roll=random.random()*5-10),
        }
        p3d_pose.move(movement)
        base_pose.move(movement)

        for bone_name in relative_pose.keys():
            assert np.isclose(base_pose.relative[bone_name].location.x,
                              p3d_pose.relative[bone_name].location.x)
            assert np.isclose(base_pose.relative[bone_name].location.y,
                              p3d_pose.relative[bone_name].location.y)
            assert np.isclose(base_pose.relative[bone_name].location.z,
                              p3d_pose.relative[bone_name].location.z)
            assert np.isclose(base_pose.relative[bone_name].rotation.pitch,
                              p3d_pose.relative[bone_name].rotation.pitch)
            assert np.isclose(base_pose.relative[bone_name].rotation.yaw,
                              p3d_pose.relative[bone_name].rotation.yaw)
            assert np.isclose(base_pose.relative[bone_name].rotation.roll,
                              p3d_pose.relative[bone_name].rotation.roll)
