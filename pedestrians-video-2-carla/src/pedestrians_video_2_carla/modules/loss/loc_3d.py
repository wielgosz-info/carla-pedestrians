from typing import Dict, Type

from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.transforms.hips_neck import (
    CarlaHipsNeckExtractor, HipsNeckNormalize)
from torch.functional import Tensor
from torch.nn.modules import loss
from pytorch_lightning.utilities.warnings import rank_zero_warn


def calculate_loss_loc_3d(criterion: loss._Loss, input_nodes: Type[CARLA_SKELETON], absolute_pose_loc: Tensor, targets: Dict[str, Tensor], **kwargs) -> Tensor:
    """
    Calculates the loss for the 3D pose.

    :param criterion: Criterion to use for the loss calculation, e.g. nn.MSELoss().
    :type criterion: _Loss
    :param input_nodes: Type of the input skeleton. For now, only CARLA_SKELETON is supported.
    :type input_nodes: Type[CARLA_SKELETON]
    :param absolute_pose_loc: Absolute pose location coordinates as calculates by the projection module.
    :type absolute_pose_loc: Tensor
    :param targets: Dictionary returned from dataset that contains the target absolute poses.
    :type targets: Dict[str, Tensor]
    :return: Calculated loss.
    :rtype: Tensor
    """
    try:
        transform = HipsNeckNormalize(CarlaHipsNeckExtractor(input_nodes))
        loss = criterion(
            transform(absolute_pose_loc, dim=3),
            transform(targets['absolute_pose_loc'], dim=3)
        )
    except AttributeError:
        rank_zero_warn('This loss is not supported for {}, only CARLA_SKELETON is supported.'.format(
            input_nodes.__name__))
        return None
    return loss
