from typing import Dict, Type
from torchmetrics import Metric
import torch
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.walker_control.torch.world import calculate_world_from_changes


class MRPE(Metric):
    """
    Mean of the Root Position Error

    It should be mainly used as a measure for the Trajectory models,
    since for the Movements root point in general is 0,0,0 before denormalization.

    Hips (extracted via HipsNeckExtractor) are considered to be a root point for this metric
    (not to be confused with CARLA skeleton root point, which is a point between feet).

    This metric is computed using 'absolute_pose_loc' ground truth and predictions to get hip points
    and 'world_loc_changes' predictions and ground truth to get the CARLA skeleton root point.
    The position error is first averaged over frames for each clip.
    Then errors are then averaged over all clips in batch. Resulting value is in millimeters.
    """

    def __init__(self,
                 dist_sync_on_step=False,
                 input_nodes: Type[CARLA_SKELETON] = CARLA_SKELETON,
                 output_nodes: Type[CARLA_SKELETON] = CARLA_SKELETON,
                 ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.pred_extractor = output_nodes.get_extractor()
        self.target_extractor = input_nodes.get_extractor()

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        try:
            prediction_pose = predictions["absolute_pose_loc"]
            target_pose = targets["absolute_pose_loc"]
            assert prediction_pose.shape == target_pose.shape

            if predictions["world_loc_changes"] is not None:
                prediction_world_loc, _ = calculate_world_from_changes(
                    predictions["world_loc_changes"].shape,
                    predictions["world_loc_changes"].device,
                    predictions["world_loc_changes"]
                )
            else:
                prediction_world_loc = predictions["world_loc"]

            target_world_loc, _ = calculate_world_from_changes(
                prediction_world_loc.shape, prediction_world_loc.device,
                targets["world_loc_changes"]
            )
            assert prediction_world_loc.shape == target_world_loc.shape

            pred_hips = self.pred_extractor.get_hips_point(prediction_pose)
            target_hips = self.target_extractor.get_hips_point(target_pose)

            world_pred_hips = prediction_world_loc + pred_hips
            world_target_hips = target_world_loc + target_hips

            avg_over_frames = torch.mean(
                torch.linalg.norm(world_pred_hips - world_target_hips, dim=-1, ord=2),
                dim=-1
            )
            self.errors += torch.sum(avg_over_frames)
            self.total += avg_over_frames.numel()
        except KeyError:
            pass

    def compute(self):
        # compute final result and convert to mm
        return 1000.0 * self.errors.float() / self.total
