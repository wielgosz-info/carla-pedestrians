"""
This file contains additional tests for pytorch3d specific implementation.
The common API tests are in tests.walker_control.test_pose.
"""

import numpy as np
import carla
import random

import torch
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


def test_batch(relative_pose, device):
    """
    Verify P3dPose.forward will work with batches as expected.
    """
    batch_size = 4

    # setup
    p3d_pose = P3dPose(device=device)
    p3d_pose.relative = relative_pose

    bones_amount = len(p3d_pose.empty)

    # get initial values
    locations = torch.empty((batch_size, bones_amount, 3), device=device)
    rotations = torch.empty((batch_size, bones_amount, 3, 3), device=device)
    for i in range(batch_size):
        (loc, rot) = p3d_pose.tensors  # tensors clones, so each call returns a copy
        locations[i] = loc
        rotations[i] = rot

    # prepare a batch of "zero" movements
    movements = torch.zeros((batch_size, bones_amount, 3), device=device)
    (abs_locations, abs_rotations, new_rotations) = p3d_pose.forward(
        movements, locations, rotations)

    # since there was not movement, all relative rotations should be as they were
    assert torch.allclose(new_rotations, rotations)

    # also, the resulting absolute transformations should be the same
    for i in range(batch_size-1):
        assert torch.allclose(abs_locations[i], abs_locations[i+1])
        assert torch.allclose(abs_rotations[i], abs_rotations[i+1])

    # add some random movements
    random_movements = torch.rand_like(movements)*5.
    (abs_locations, abs_rotations, new_rotations) = p3d_pose.forward(
        random_movements, locations, rotations)

    # also, the resulting locations/rotations should be all different
    for i in range(batch_size-1):
        assert not torch.allclose(abs_locations[i], abs_locations[i+1])
        assert not torch.allclose(abs_rotations[i], abs_rotations[i+1])
        assert not torch.allclose(new_rotations[i], new_rotations[i+1])
