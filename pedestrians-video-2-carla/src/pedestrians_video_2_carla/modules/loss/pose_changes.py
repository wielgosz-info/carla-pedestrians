from typing import Dict

from torch.functional import Tensor
from torch.nn.modules import loss


def calculate_loss_pose_changes(criterion: loss._Loss, pose_inputs: Tensor, targets: Dict[str, Tensor], **kwargs) -> Tensor:
    """
    Calculates the loss by directly comparing expected pose changes with the target pose changes.
    It doesn't really work well for 'sparse random twitches', since the target pose changes are
    mostly 0s or occasionally a few small (of 1e-2 order) numbers.

    :param criterion: Criterion to use for the loss calculation, e.g. nn.MSELoss().
    :type criterion: _Loss
    :param pose_inputs: The pose changes to compare with the target pose changes.
    :type pose_inputs: Tensor
    :param targets: Dictionary returned from dataset that containins the target pose changes.
    :type targets: Dict[str, Tensor]
    :return: Calculated loss.
    :rtype: Tensor
    """

    loss = criterion(
        pose_inputs,
        targets['pose_inputs']
    )

    return loss
