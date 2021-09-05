import pytest

import carla

from pedestrians_video_2_carla.walker_control.io import load_reference
from pedestrians_video_2_carla.walker_control.transforms import unreal_to_carla
from pedestrians_video_2_carla.walker_control.pose import Pose


def test_set_pose():
    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    relative_pose = unreal_to_carla(unreal_rel_pose['transforms'])

    p = Pose()
    p.relative = relative_pose

    for bone_name, transforms_dict in relative_pose.items():
        assert p.relative[bone_name] == carla.Transform(
            location=carla.Location(
                x=transforms_dict['location']['x'],
                y=transforms_dict['location']['y'],
                z=transforms_dict['location']['z'],
            ),
            rotation=carla.Rotation(
                pitch=transforms_dict['rotation']['pitch'],
                yaw=transforms_dict['rotation']['yaw'],
                roll=transforms_dict['rotation']['roll'],
            )
        )


def test_relative_to_absolute():
    unreal_abs_pose = load_reference('sk_female_absolute.yaml')
    absolute_pose = unreal_to_carla(unreal_abs_pose['transforms'])

    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    relative_pose = unreal_to_carla(unreal_rel_pose['transforms'])

    p = Pose()
    p.relative = relative_pose

    for bone_name, transforms_dict in absolute_pose.items():
        assert p.absolute[bone_name] == carla.Transform(
            location=carla.Location(
                x=transforms_dict['location']['x'],
                y=transforms_dict['location']['y'],
                z=transforms_dict['location']['z'],
            ),
            rotation=carla.Rotation(
                pitch=transforms_dict['rotation']['pitch'],
                yaw=transforms_dict['rotation']['yaw'],
                roll=transforms_dict['rotation']['roll'],
            )
        )
