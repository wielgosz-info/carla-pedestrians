import numpy as np
import torch
import carla
import random
from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.utils.unreal import load_reference, unreal_to_carla
from pedestrians_video_2_carla.walker_control.pose import Pose


def test_set_get_pose():
    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    relative_pose = unreal_to_carla(unreal_rel_pose['transforms'])

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # set
    p = P3dPose(device=device)
    p.relative = relative_pose

    # get
    relative = p.relative

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


def test_relative_to_absolute():
    unreal_abs_pose = load_reference('sk_female_absolute.yaml')
    absolute_pose = unreal_to_carla(unreal_abs_pose['transforms'])

    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    relative_pose = unreal_to_carla(unreal_rel_pose['transforms'])

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # set
    p = P3dPose(device=device)
    p.relative = relative_pose

    # get
    absolute = p.absolute

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


def test_move():
    """
    Tests movement calculations by comparing them with "base" Pose implementation
    """

    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    relative_pose = unreal_to_carla(unreal_rel_pose['transforms'])

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

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
