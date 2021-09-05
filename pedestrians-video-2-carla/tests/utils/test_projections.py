from pedestrians_video_2_carla.utils.openpose import (
    load_openpose, openpose_to_projection_points)
from pedestrians_video_2_carla.utils.projections import \
    scale_projections_by_height
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.pose_projection import \
    PoseProjection


def test_scaling():
    pedestrian = ControlledPedestrian(None, 'adult', 'female')

    projection = PoseProjection(None, pedestrian)
    image_projection_points = projection.current_pose_to_points()

    raw_openpose = load_openpose(
        '/app/docs/_static/images/reference_pose/openpose_keypoints.json')
    image_openpose_points = openpose_to_projection_points(
        raw_openpose[0][0]['pose_keypoints_2d'], pedestrian.current_pose.empty)

    (projection_points, openpose_points) = scale_projections_by_height(
        image_projection_points,
        image_openpose_points,
        pedestrian.current_pose.empty
    )

    assert projection_points[:, 1].min() == 0.0
    assert projection_points[:, 1].max() == 1.0
    assert openpose_points[:, 1].min() == 0.0
    assert openpose_points[:, 1].max() == 1.0
