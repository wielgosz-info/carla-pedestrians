from typing import Dict

import torch
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from torch.functional import Tensor
from torch.nn.modules import loss


def calculate_loss_cum_pose_changes(criterion: loss._Loss, pose_changes: Tensor, targets: Dict[str, Tensor], **kwargs) -> Tensor:
    """
    Calculates the loss by comparing the pose changes accumulated over the frames.

    :param criterion: Criterion to use for the loss calculation, e.g. nn.MSELoss().
    :type criterion: _Loss
    :param pose_changes: The pose changes to compare with the target pose changes.
    :type pose_changes: Tensor
    :param targets: Dictionary returned from dataset that containins the target pose changes.
    :type targets: Dict[str, Tensor]
    :return: Calculated loss.
    :rtype: Tensor
    """

    # calculate cumulative rotations
    (batch_size, clip_length, bones, *_) = pose_changes.shape

    cumulative_changes = []
    cumulative_targets = []

    prev_changes = torch.eye(3, device=pose_changes.device).reshape(
        (1, 3, 3)).repeat((batch_size*bones, 1, 1))
    prev_targets = torch.eye(3, device=pose_changes.device).reshape(
        (1, 3, 3)).repeat((batch_size*bones, 1, 1))

    matrix_pose_changes = euler_angles_to_matrix(pose_changes, "XYZ")
    matrix_targets = euler_angles_to_matrix(targets['pose_changes'], "XYZ")

    for i in range(clip_length):
        prev_changes = torch.bmm(
            prev_changes,
            matrix_pose_changes[:, i].reshape((-1, 3, 3))
        )
        prev_targets = torch.bmm(
            prev_targets,
            matrix_targets[:, i].reshape((-1, 3, 3))
        )
        cumulative_changes.append(prev_changes.reshape((batch_size, bones, 3, 3)))
        cumulative_targets.append(prev_targets.reshape((batch_size, bones, 3, 3)))

    loss = criterion(
        torch.stack(cumulative_changes, dim=1),
        torch.stack(cumulative_targets, dim=1),
    )

    return loss
