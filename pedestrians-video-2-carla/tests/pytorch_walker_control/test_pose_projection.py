"""
This file contains additional tests for pytorch3d specific implementation.
The common API tests are in tests.walker_control.test_pose_projection.
"""

import carla
import numpy as np
import torch

from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.pytorch_walker_control.pose_projection import \
    P3dPoseProjection
from pedestrians_video_2_carla.walker_control.pose_projection import \
    PoseProjection
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian


def test_p3d_pose_projection_matches_base_pose_projection(device, pedestrian):
    base_projection = PoseProjection(pedestrian, None)
    p3d_projection = P3dPoseProjection(device, pedestrian, None)

    base_points = base_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    # base_projection.current_pose_to_image('reference_base_1', base_points)
    # p3d_projection.current_pose_to_image('reference_pytorch3d_1', p3d_points)
    assert np.allclose(base_points, p3d_points)

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(0.5, 0, 0),
    ))
    pedestrian.update_pose({
        'crl_arm__L': carla.Rotation(yaw=-30),
        'crl_foreArm__L': carla.Rotation(pitch=-30)
    })
    base_points = base_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    # base_projection.current_pose_to_image('reference_base_2', base_points)
    # p3d_projection.current_pose_to_image('reference_pytorch3d_2', p3d_points)
    assert np.allclose(base_points, p3d_points)

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(0, 0.5, 0),
    ))
    base_points = base_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    # base_projection.current_pose_to_image('reference_base_3', base_points)
    # p3d_projection.current_pose_to_image('reference_pytorch3d_3', p3d_points)
    assert np.allclose(base_points, p3d_points)

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(0, 0, 0.5)
    ))
    base_points = base_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    # base_projection.current_pose_to_image('reference_base_4', base_points)
    # p3d_projection.current_pose_to_image('reference_pytorch3d_4', p3d_points)
    assert np.allclose(base_points, p3d_points)

    pedestrian.teleport_by(carla.Transform(
        rotation=carla.Rotation(yaw=30)
    ))
    base_points = base_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    # base_projection.current_pose_to_image('reference_base_5', base_points)
    # p3d_projection.current_pose_to_image('reference_pytorch3d_5', p3d_points)
    assert np.allclose(base_points, p3d_points)


def test_batch(device):
    # we only use pedestrian instance to get initial values,
    # the remembered pose will NOT (and should not!) be updated during this test
    # since only `.forward` methods will be used

    pedestrian = ControlledPedestrian(
        None, 'adult', 'female', pose_cls=P3dPose, device=device)

    p3d_pose = pedestrian.current_pose
    p3d_projection = P3dPoseProjection(device, pedestrian, None)

    batch_size = 4
    bones_amount = len(p3d_pose.empty)

    # get initial values
    locations = torch.empty((batch_size, bones_amount, 3), device=device)
    rotations = torch.empty((batch_size, bones_amount, 3, 3), device=device)
    world_loc = torch.zeros((batch_size, 3), device=device)
    world_rot = torch.zeros((batch_size, 3), device=device)
    for i in range(batch_size):
        (loc, rot) = p3d_pose.tensors  # 'tensors' is cloning, so each call returns a copy
        locations[i] = loc
        rotations[i] = rot

    # prepare a batch of "zero" movements
    movements = torch.zeros((batch_size, bones_amount, 3), device=device)
    (abs_locations, _, _) = p3d_pose.forward(
        movements, locations, rotations)

    # feed the results into pose projection
    p3d_points = p3d_projection.forward(abs_locations, world_loc, world_rot)

    # ensure we've got a batch of projections
    assert p3d_points.shape[0] == batch_size

    # all results should be the same
    for i in range(batch_size-1):
        assert torch.allclose(p3d_points[i], p3d_points[i+1])
        assert torch.allclose(p3d_points[i], p3d_points[i+1])

    # add some random movements
    random_movements = torch.rand_like(movements)*5.
    (abs_locations, _, _) = p3d_pose.forward(
        random_movements, locations, rotations)

    p3d_points = p3d_projection.forward(abs_locations, world_loc, world_rot)

    # all results should be different
    for i in range(batch_size-1):
        assert not torch.allclose(p3d_points[i], p3d_points[i+1])
        assert not torch.allclose(p3d_points[i], p3d_points[i+1])
