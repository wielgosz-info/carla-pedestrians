from typing import Dict
from torchmetrics import Metric
import torch
from pedestrians_video_2_carla.submodules.video_pose_3d.loss import n_mpjpe


class FB_N_MPJPE(Metric):
    """
    Normalized Mean Per Joint Position Error (scale only).
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

            frames_num = torch.prod(torch.tensor(original_shape[:-2]))
            metric = n_mpjpe(prediction, target)

            self.errors += frames_num * metric
            self.total += frames_num
        except KeyError:
            pass

    def compute(self):
        # compute final result and convert to mm
        return 1000.0 * self.errors.float() / self.total
