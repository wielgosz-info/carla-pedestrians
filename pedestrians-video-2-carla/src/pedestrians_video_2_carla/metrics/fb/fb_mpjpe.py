from typing import Dict
from torchmetrics import Metric
import torch
from pedestrians_video_2_carla.submodules.video_pose_3d.loss import mpjpe


class FB_MPJPE(Metric):
    """
    Mean Per Joint Position Error.
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        try:
            prediction = predictions["absolute_pose_loc"]
            target = targets["absolute_pose_loc"]

            original_shape = prediction.shape
            prediction = prediction.view((-1, *original_shape[-2:]))
            target = target.view((-1, *original_shape[-2:]))

            frames_num = torch.prod(torch.tensor(original_shape[:-2]))
            metric = mpjpe(prediction, target)

            self.errors += frames_num * metric
            self.total += frames_num
        except KeyError:
            pass

    def compute(self):
        # compute final result and convert to mm
        return 1000.0 * self.errors.float() / self.total
