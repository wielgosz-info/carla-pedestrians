from typing import Dict

from torch.functional import Tensor
from torch.nn.modules import loss


def calculate_loss_rot_3d(criterion: loss._Loss, absolute_pose_rot: Tensor, targets: Dict[str, Tensor], **kwargs) -> Tensor:
    """
    Calculates the loss for the 3D pose.

    :param criterion: Criterion to use for the loss calculation, e.g. nn.MSELoss().
    :type criterion: _Loss
    :param absolute_pose_rot: Absolute pose rotation coordinates as calculates by the projection module.
    :type absolute_pose_rot: Tensor
    :param targets: Dictionary returned from dataset that contains the target absolute poses.
    :type targets: Dict[str, Tensor]
    :return: Calculated loss.
    :rtype: Tensor
    """

    if absolute_pose_rot is None or 'absolute_pose_rot' not in targets:
        return None

    loss = criterion(
        absolute_pose_rot,
        targets['absolute_pose_rot']
    )

    return loss
