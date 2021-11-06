from typing import Type

from pedestrians_video_2_carla.skeletons.nodes import MAPPINGS, Skeleton
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from torch.functional import Tensor
from torch.nn.modules import loss


def get_common_indices(input_nodes: Type[Skeleton] = CARLA_SKELETON):
    if input_nodes == CARLA_SKELETON:
        carla_indices = slice(None)
        input_indices = slice(None)
    else:
        mappings = MAPPINGS[input_nodes]
        (carla_indices, input_indices) = zip(
            *[(c.value, o.value) for (c, o) in mappings])

    return carla_indices, input_indices


def calculate_loss_common_loc_2d(criterion: loss._Loss, input_nodes: Type[Skeleton], normalized_projection: Tensor, frames: Tensor, **kwargs) -> Tensor:
    """
    Calculates the loss for the 2D pose projection.
    Only accounts for common nodes between input skeleton and CARLA_SKELETON, as defined in MAPPINGS.

    :param criterion: Criterion to use for the loss calculation, e.g. nn.MSELoss().
    :type criterion: _Loss
    :param input_nodes: Type of the input skeleton, e.g. BODY_25_SKELETON or CARLA_SKELETON.
    :type input_nodes: Type[Skeleton]
    :param normalized_projection: Normalized projection as calculated by the projection module.
    :type normalized_projection: Tensor
    :param frames: Model input frames, containing the normalized inputs 2D projection.
    :type frames: Tensor
    :return: Calculated loss.
    :rtype: Tensor
    """
    carla_indices, input_indices = get_common_indices(input_nodes)

    common_projection = normalized_projection[..., carla_indices, 0:2]
    common_input = frames[..., input_indices, 0:2]

    loss = criterion(
        common_projection,
        common_input
    )

    return loss
