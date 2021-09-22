"""
This file contains additional tests for pytorch3d specific implementation.
The common API tests are in tests.walker_control.test_pose_projection.
"""

import carla
import numpy as np

from pedestrians_video_2_carla.pytorch_walker_control.pose_projection import \
    P3dPoseProjection
from pedestrians_video_2_carla.walker_control.pose_projection import \
    PoseProjection


def test_p3d_pose_projection_matches_base_pose_projection(device, pedestrian):
    base_projection = PoseProjection(pedestrian, None)
    p3d_projection = P3dPoseProjection(device, pedestrian, None)

    base_points = base_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    base_projection.current_pose_to_image('reference_base_1', base_points)
    p3d_projection.current_pose_to_image('reference_pytorch3d_1', p3d_points)
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
    base_projection.current_pose_to_image('reference_base_2', base_points)
    p3d_projection.current_pose_to_image('reference_pytorch3d_2', p3d_points)
    assert np.allclose(base_points, p3d_points)

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(0, 0.5, 0),
    ))
    base_points = base_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    base_projection.current_pose_to_image('reference_base_3', base_points)
    p3d_projection.current_pose_to_image('reference_pytorch3d_3', p3d_points)
    assert np.allclose(base_points, p3d_points)

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(0, 0, 0.5)
    ))
    base_points = base_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    base_projection.current_pose_to_image('reference_base_4', base_points)
    p3d_projection.current_pose_to_image('reference_pytorch3d_4', p3d_points)
    assert np.allclose(base_points, p3d_points)

    pedestrian.teleport_by(carla.Transform(
        rotation=carla.Rotation(yaw=30)
    ))
    base_points = base_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    base_projection.current_pose_to_image('reference_base_5', base_points)
    p3d_projection.current_pose_to_image('reference_pytorch3d_5', p3d_points)
    assert np.allclose(base_points, p3d_points)
