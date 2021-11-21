from enum import Enum

from torch import nn

from pedestrians_video_2_carla.modules.base.output_types import MovementsModelOutputType

from .pose_changes import calculate_loss_pose_changes
from .loc_3d import calculate_loss_loc_3d
from .common_loc_2d import calculate_loss_common_loc_2d
from .loc_2d_3d import calculate_loss_loc_2d_3d
from .cum_pose_changes import calculate_loss_cum_pose_changes
from .rot_3d import calculate_loss_rot_3d
from .loc_2d_loc_rot_3d import calculate_loss_loc_2d_loc_rot_3d


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

    # Complex losses depending on base losses
    # Do NOT declare a complex loss depending on another complex loss,
    # it will most likely not work since there is no complex
    # dependencies resolving done, only "lets put all the dependencies first,
    # and actual losses later in the order of calculations".
    loc_2d_3d = (calculate_loss_loc_2d_3d, None, [
        'common_loc_2d', 'loc_3d'
    ])
    loc_2d_loc_rot_3d = (calculate_loss_loc_2d_loc_rot_3d, None, [
        'common_loc_2d', 'loc_3d', 'rot_3d'
    ])

    @staticmethod
    def get_supported_loss_modes(movements_output_type: MovementsModelOutputType):
        """
        Returns a list of supported loss modes for a given projection type.
        """
        return {
            MovementsModelOutputType.pose_changes: list(LossModes),
            MovementsModelOutputType.absolute_loc_rot: [
                LossModes.common_loc_2d,
                LossModes.loc_3d,
                LossModes.rot_3d,
                LossModes.loc_2d_3d,
                LossModes.loc_2d_loc_rot_3d,
            ],
            MovementsModelOutputType.absolute_loc: [
                LossModes.common_loc_2d,
                LossModes.loc_3d,
                LossModes.loc_2d_3d,
            ]
        }[movements_output_type]
