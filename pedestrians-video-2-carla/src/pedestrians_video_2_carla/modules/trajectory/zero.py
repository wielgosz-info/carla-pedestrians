import torch
from pedestrians_video_2_carla.modules.base.trajectory import TrajectoryModel


class ZeroTrajectory(TrajectoryModel):
    """
    Dummy module that is not changing pedestrian world position & orientation at all.
    """

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        return (
            # world location changes
            torch.zeros((*original_shape[:2], 3), device=x.device),
            torch.eye(3, device=x.device).reshape((1, 1, 3, 3)).repeat(
                (*original_shape[:2], 1, 1)),  # world rotation changes
        )

    def configure_optimizers(self):
        return {}
