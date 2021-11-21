from typing import Tuple
import torch
from torch.functional import Tensor


def calculate_world_from_changes(
    shape: Tuple,
    device: torch.device,
    world_loc_change_batch: Tensor = None,
    world_rot_change_batch: Tensor = None,
    initial_world_loc: Tensor = None,
    initial_world_rot: Tensor = None,
) -> Tuple[Tensor, Tensor]:
    batch_size, clip_length, *_ = shape

    if initial_world_loc is None:
        initial_world_loc = torch.zeros((batch_size, 3),
                                        device=device)  # zero world loc

    if initial_world_rot is None:
        initial_world_rot = torch.eye(3, device=device).reshape(
            (1, 3, 3)).repeat((batch_size, 1, 1))  # zero world rot

    if world_loc_change_batch is None and world_rot_change_batch is None:
        # bail out if no changes
        return (
            initial_world_loc.unsqueeze(1).repeat(1, clip_length, 1),
            initial_world_rot.unsqueeze(1).repeat(1, clip_length, 1, 1),
        )

    if world_loc_change_batch is None:
        world_loc_change_batch = torch.zeros((batch_size, clip_length, 3),
                                             device=device)  # no world loc change

    if world_rot_change_batch is None:
        world_rot_change_batch = torch.eye(3, device=device).reshape(
            (1, 1, 3, 3)).repeat((batch_size, clip_length, 1, 1))  # no world rot change

    world_loc = torch.empty(
        (batch_size, clip_length+1, *world_loc_change_batch.shape[2:]), device=device)
    world_rot = torch.empty(
        (batch_size, clip_length+1, *world_rot_change_batch.shape[2:]), device=device)

    world_loc[:, 0] = initial_world_loc
    world_rot[:, 0] = initial_world_rot

    # for every frame in clip
    for i in range(clip_length):
        world_rot[:, i+1] = torch.bmm(
            world_rot[:, i],
            world_rot_change_batch[:, i]
        )
        world_loc[:, i+1] = world_loc[:, i] + world_loc_change_batch[:, i]

    return world_loc[:, 1:], world_rot[:, 1:]
