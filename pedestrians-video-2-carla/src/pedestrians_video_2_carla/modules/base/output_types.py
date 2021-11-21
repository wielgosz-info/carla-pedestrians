from enum import Enum


class MovementsModelOutputType(Enum):
    """
    Enum for the different model types.
    """
    pose_changes = 0  # default, prefferred

    # undesired, but possible; it will most likely deform the skeleton; incompatible with some loss functions
    absolute_loc_rot = 1

    # undesired, but possible; it will most likely deform the skeleton and results in broken rotations; incompatible with some loss functions
    absolute_loc = 2


class TrajectoryModelOutputType(Enum):
    """
    Enum for the different model types.
    """
    changes = 0  # default
    loc_rot = 1
