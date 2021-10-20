from typing import Union
import torch
from torch import nn

from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO, COMMON_NODES
from pedestrians_video_2_carla.pytorch_helpers.transforms import CarlaHipsNeckNormalize


class ProjectionLoss(nn.Module):
    def __init__(self, pose_projection_func, points: Union[BODY_25, COCO] = BODY_25, criterion=None, projection_transform=None) -> None:
        super().__init__()

        self.pose_projection = pose_projection_func
        self.__points = points

        if criterion is None:
            criterion = nn.MSELoss(reduction='mean')
        self.criterion = criterion

        if projection_transform is None:
            projection_transform = CarlaHipsNeckNormalize()
        self.projection_transform = projection_transform

    def forward(self, inputs, targets):
        projected_pose = self.pose_projection(
            inputs,
            torch.zeros((*targets.shape[:2], 3),
                        device=targets.device),  # no world loc change
            torch.zeros((*targets.shape[:2], 3),
                        device=targets.device),  # no world rot change
        )
        normalized_projection = self.projection_transform(projected_pose)

        if type(self.__points) == COCO:
            mappings = COMMON_NODES['CARLA_2_COCO']
        else:  # default
            mappings = COMMON_NODES['CARLA_2_BODY_25']

        (carla_indices, openpose_indices) = zip(
            *[(c.value, o.value) for (c, o) in mappings])

        common_projection = normalized_projection[..., carla_indices, 0:2]
        common_openpose = targets[..., openpose_indices, 0:2]

        loss = self.criterion(
            common_projection,
            common_openpose
        )

        return loss
