import torch
from pedestrians_video_2_carla.modules.base.movements import MovementsModel


class ZeroMovements(MovementsModel):
    """
    Dummy module that is not changing pedestrian skeleton at all.
    """

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        return torch.zeros((*original_shape[:3], 3), device=x.device)  # 3D pose changes

    def configure_optimizers(self):
        return {}
