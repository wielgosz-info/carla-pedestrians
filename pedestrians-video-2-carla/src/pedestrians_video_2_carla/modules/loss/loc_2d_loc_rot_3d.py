from typing import Dict

from torch.functional import Tensor


def calculate_loss_loc_2d_loc_rot_3d(requirements: Dict[str, Tensor], **kwargs) -> Tensor:
    """
    Calculates the simple sum of the 'common_loc_2d', 'loc_3d' and 'rot_3d' losses.

    :param requirements: The dictionary containing the calculated 'common_loc_2d', 'loc_3d', and 'rot_3d'.
    :type requirements: Dict[str, Tensor]
    :return: The sum of the 'common_loc_2d', 'loc_3d', and 'rot_3d' losses.
    :rtype: Tensor
    """
    loss = requirements['common_loc_2d'] + \
        requirements['loc_3d'] + requirements['rot_3d']

    return loss
