from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.utils.unreal import load_reference, unreal_to_carla


def test_relative_to_absolute():
    unreal_abs_pose = load_reference('sk_female_absolute.yaml')
    absolute_pose = unreal_to_carla(unreal_abs_pose['transforms'])

    p = P3dPose()
