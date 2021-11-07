from enum import Enum

from torch import nn

from .pose_changes import calculate_loss_pose_changes
from .loc_3d import calculate_loss_loc_3d
from .common_loc_2d import calculate_loss_common_loc_2d
from .loc_2d_3d import calculate_loss_loc_2d_3d
from .cum_pose_changes import calculate_loss_cum_pose_changes
from .rot_3d import calculate_loss_rot_3d


class LossModes(Enum):
    """
    Enum for loss modes.

    For now this will not work with ddp_spawn, because it is not picklable, use ddp accelerator instead.
    """
    # Base functions with MSE
    common_loc_2d = (calculate_loss_common_loc_2d, nn.MSELoss(reduction='mean'))
    loc_3d = (calculate_loss_loc_3d, nn.MSELoss(reduction='mean'))
    rot_3d = (calculate_loss_rot_3d, nn.MSELoss(reduction='mean'))
    cum_pose_changes = (calculate_loss_cum_pose_changes, nn.MSELoss(reduction='mean'))
    pose_changes = (calculate_loss_pose_changes, nn.MSELoss(reduction='sum'))

    # Complex loss depending on other losses
    loc_2d_3d = (calculate_loss_loc_2d_3d, None, [
        'common_loc_2d', 'loc_3d'
    ])

    def __init__(self, loss_fn, criterion, dependencies=None):
        self.loss_fn = loss_fn
        self.criterion = criterion
        self.dependencies = dependencies

    def __hash__(self):
        return hash(self.name)
