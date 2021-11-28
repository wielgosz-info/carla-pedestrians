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

        self.w = w if w is not None else torch.ones((1, 1, 1, 1))

        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        try:
            prediction = predictions["absolute_pose_loc"]
            target = targets["absolute_pose_loc"]

            weights = self.w
            if weights.shape[0] != prediction.shape[0]:
                weights = weights.repeat(prediction.shape[0], 1, 1, 1)
            weights = weights.to(prediction.device)

            frames_num = prediction.shape[0] * prediction.shape[1]
            metric = weighted_mpjpe(prediction, target, weights)

            self.errors += frames_num * metric
            self.total += frames_num
        except KeyError:
            pass

    def compute(self):
        # compute final result and convert to mm
        return 1000.0 * self.errors.float() / self.total
