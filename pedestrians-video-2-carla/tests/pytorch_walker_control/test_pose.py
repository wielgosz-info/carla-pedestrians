import numpy as np
import torch
from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.utils.unreal import load_reference, unreal_to_carla


def test_set_pose():
    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    relative_pose = unreal_to_carla(unreal_rel_pose['transforms'])

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    p = P3dPose(device=device)
    p.relative = relative_pose

    for bone_name, transforms_dict in relative_pose.items():
        assert np.isclose(p.relative[bone_name].location.x, transforms_dict.location.x)
        assert np.isclose(p.relative[bone_name].location.y, transforms_dict.location.y)
        assert np.isclose(p.relative[bone_name].location.z, transforms_dict.location.z)
        assert np.isclose(p.relative[bone_name].rotation.pitch,
                          transforms_dict.rotation.pitch)
        assert np.isclose(p.relative[bone_name].rotation.yaw,
                          transforms_dict.rotation.yaw)
        assert np.isclose(p.relative[bone_name].rotation.roll,
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

    p = P3dPose(device=device)
    p.relative = relative_pose
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
