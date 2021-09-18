import carla
import numpy as np
import torch
from pedestrians_video_2_carla.pytorch_carla_pedestrians.pose_projection import \
    PoseProjection
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.pose_projection import \
    PoseProjection as CTPoseProjection


def test_p3d_pose_projection_matches_ct_pose_projection():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    pedestrian = ControlledPedestrian(None, 'adult', 'female')
    ct_projection = CTPoseProjection(pedestrian, None)
    p3d_projection = PoseProjection(device, pedestrian, None)

    ct_points = ct_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    ct_projection.current_pose_to_image('reference_ct', ct_points)
    p3d_projection.current_pose_to_image('reference_pytorch3d', p3d_points)
    assert np.allclose(ct_points, p3d_points.cpu().numpy()[..., :2])

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(0.5, 0, 0),
    ))
    pedestrian.update_pose({
        'crl_arm__L': carla.Rotation(yaw=-30),
        'crl_foreArm__L': carla.Rotation(pitch=-30)
    })
    ct_points = ct_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    ct_projection.current_pose_to_image('reference_ct_2', ct_points)
    p3d_projection.current_pose_to_image('reference_pytorch3d_2', p3d_points)
    assert np.allclose(ct_points, p3d_points.cpu().numpy()[..., :2])

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(0, 0.5, 0),
    ))
    ct_points = ct_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    ct_projection.current_pose_to_image('reference_ct_3', ct_points)
    p3d_projection.current_pose_to_image('reference_pytorch3d_3', p3d_points)
    assert np.allclose(ct_points, p3d_points.cpu().numpy()[..., :2])

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(0, 0, 0.5)
    ))
    ct_points = ct_projection.current_pose_to_points()
    p3d_points = p3d_projection.current_pose_to_points()
    ct_projection.current_pose_to_image('reference_ct_4', ct_points)
    p3d_projection.current_pose_to_image('reference_pytorch3d_4', p3d_points)
    assert np.allclose(ct_points, p3d_points.cpu().numpy()[..., :2])
