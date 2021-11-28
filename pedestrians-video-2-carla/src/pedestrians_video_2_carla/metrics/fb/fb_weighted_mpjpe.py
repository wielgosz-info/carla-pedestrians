from typing import Dict
from torchmetrics import Metric
import torch
from pedestrians_video_2_carla.submodules.video_pose_3d.loss import weighted_mpjpe


class FB_WeightedMPJPE(Metric):
    """
    Weighted Mean Per Joint Position Error.
    """

    def __init__(self, dist_sync_on_step=False, w=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.w = w

        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        try:
            prediction = predictions["absolute_pose_loc"]
            target = targets["absolute_pose_loc"]

            original_shape = prediction.shape
            frames_num = torch.prod(torch.tensor(original_shape[:-2]))

            if self.w is not None:
                weights = self.w
            else:
                weights = torch.ones((1, 1, *original_shape[-2:-1]))
            if weights.shape[0] != frames_num:
                weights = weights.repeat((*original_shape[:-2], 1))
            weights = weights.to(prediction.device)

            metric = weighted_mpjpe(prediction, target, weights)

            self.errors += frames_num * metric
            self.total += frames_num
        except KeyError:
            pass

    def compute(self):
        # compute final result and convert to mm
        return 1000.0 * self.errors.float() / self.total
