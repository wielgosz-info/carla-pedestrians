from pedestrians_video_2_carla.walker_control.torch.world import calculate_world_from_changes
import torch
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix


def test_calculate_world_from_changes_defaults(device):
    batch_size = 2
    clip_length = 10

    world_loc, world_rot = calculate_world_from_changes(
        (batch_size, clip_length, 1, 1), device)

    assert world_loc.shape == (batch_size, clip_length,
                               3), "World location has incorrect shape"
    assert world_rot.shape == (batch_size, clip_length, 3,
                               3), "World rotation has incorrect shape"

    gt_world_loc = torch.zeros((batch_size, clip_length, 3), device=device)
    gt_world_rot = torch.eye(3, device=device).reshape(
        (1, 1, 3, 3)).repeat((batch_size, clip_length, 1, 1))

    assert torch.allclose(world_loc, gt_world_loc), "World location is not zero"
    assert torch.allclose(world_rot, gt_world_rot), "World rotation is not zero"


def test_calculate_world_from_zero_changes(device):
    batch_size = 2
    clip_length = 10

    zero_world_loc_changes = torch.zeros((batch_size, clip_length, 3), device=device)
    zero_world_rot_changes = torch.eye(3, device=device).reshape(
        (1, 1, 3, 3)).repeat((batch_size, clip_length, 1, 1))

    world_loc, world_rot = calculate_world_from_changes(
        (batch_size, clip_length, 1, 1), device,
        world_loc_change_batch=zero_world_loc_changes,
        world_rot_change_batch=zero_world_rot_changes
    )

    assert world_loc.shape == (batch_size, clip_length,
                               3), "World location has incorrect shape"
    assert world_rot.shape == (batch_size, clip_length, 3,
                               3), "World rotation has incorrect shape"

    assert torch.allclose(
        world_loc, zero_world_loc_changes), "World location is not zero"
    assert torch.allclose(
        world_rot, zero_world_rot_changes), "World rotation is not zero"


def test_calculate_world_from_changes_defaults_and_initial(device):
    batch_size = 2
    clip_length = 10

    gt_world_loc = torch.rand((batch_size, 3), device=device)
    gt_world_rot = euler_angles_to_matrix(torch.rand(
        (batch_size, 3), device=device), "XYZ")

    world_loc, world_rot = calculate_world_from_changes(
        (batch_size, clip_length, 1, 1), device,
        initial_world_loc=gt_world_loc,
        initial_world_rot=gt_world_rot
    )

    assert world_loc.shape == (batch_size, clip_length,
                               3), "World location has incorrect shape"
    assert world_rot.shape == (batch_size, clip_length, 3,
                               3), "World rotation has incorrect shape"

    assert torch.allclose(
        world_loc[:, 0], gt_world_loc), "World location does not match initial location"
    assert torch.allclose(
        world_rot[:, 0], gt_world_rot), "World rotation does not match initial rotation"

    assert torch.allclose(
        world_loc[:, -1], gt_world_loc), "World location does not match initial location"
    assert torch.allclose(
        world_rot[:, -1], gt_world_rot), "World rotation does not match initial rotation"


def test_calculate_world_from_zero_changes_and_initial(device):
    batch_size = 2
    clip_length = 10

    gt_world_loc = torch.rand((batch_size, 3), device=device)
    gt_world_rot = euler_angles_to_matrix(torch.rand(
        (batch_size, 3), device=device), "XYZ")

    zero_world_loc_changes = torch.zeros((batch_size, clip_length, 3), device=device)
    zero_world_rot_changes = torch.eye(3, device=device).reshape(
        (1, 1, 3, 3)).repeat((batch_size, clip_length, 1, 1))

    world_loc, world_rot = calculate_world_from_changes(
        (batch_size, clip_length, 1, 1), device,
        world_loc_change_batch=zero_world_loc_changes,
        world_rot_change_batch=zero_world_rot_changes,
        initial_world_loc=gt_world_loc,
        initial_world_rot=gt_world_rot
    )

    assert world_loc.shape == (batch_size, clip_length,
                               3), "World location has incorrect shape"
    assert world_rot.shape == (batch_size, clip_length, 3,
                               3), "World rotation has incorrect shape"

    assert torch.allclose(
        world_loc[:, 0], gt_world_loc), "World location does not match initial location"
    assert torch.allclose(
        world_rot[:, 0], gt_world_rot), "World rotation does not match initial rotation"

    assert torch.allclose(
        world_loc[:, -1], gt_world_loc), "World location does not match initial location"
    assert torch.allclose(
        world_rot[:, -1], gt_world_rot), "World rotation does not match initial rotation"
