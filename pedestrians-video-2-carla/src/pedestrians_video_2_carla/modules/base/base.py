
import platform
from typing import Any, List, Tuple, Type

import pytorch_lightning as pl
import torch
from pedestrians_video_2_carla.metrics.fb import *
from pedestrians_video_2_carla.metrics.mpjpe import MPJPE
from pedestrians_video_2_carla.metrics.mrpe import MRPE
from pedestrians_video_2_carla.modules.base.movements import MovementsModel
from pedestrians_video_2_carla.modules.base.output_types import (
    MovementsModelOutputType, TrajectoryModelOutputType)
from pedestrians_video_2_carla.modules.base.trajectory import TrajectoryModel
from pedestrians_video_2_carla.modules.layers.projection import \
    ProjectionModule
from pedestrians_video_2_carla.modules.loss import LossModes
from pedestrians_video_2_carla.modules.movements.zero import ZeroMovements
from pedestrians_video_2_carla.modules.trajectory.zero import ZeroTrajectory
from pedestrians_video_2_carla.skeletons.nodes import get_skeleton_type_by_name
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pytorch_lightning.utilities import rank_zero_only
from torch.functional import Tensor
from torchmetrics import MetricCollection


class LitBaseMapper(pl.LightningModule):
    """
    Base LightningModule - all other LightningModules should inherit from this.
    It contains movements model, trajectory model, projection layer, loss modes handling and video logging.

    Movements & Trajectory models should implement forward() and configure_optimizers() method.
    If they use additional hyperparameters, they should also set self._hparams dict
    in __init__() and (optionally) override add_model_specific_args() method.
    """

    def __init__(
        self,
        movements_model: MovementsModel = None,
        trajectory_model: TrajectoryModel = None,
        loss_modes: List[LossModes] = None,
        **kwargs
    ):
        super().__init__()

        # default layers
        if movements_model is None:
            movements_model = ZeroMovements()
        self.movements_model = movements_model

        if trajectory_model is None:
            trajectory_model = ZeroTrajectory()
        self.trajectory_model = trajectory_model

        self.projection = ProjectionModule(
            movements_output_type=self.movements_model.output_type,
            trajectory_output_type=self.trajectory_model.output_type,
        )

        # losses
        if loss_modes is None or len(loss_modes) == 0:
            loss_modes = [LossModes.common_loc_2d]
        self._loss_modes = loss_modes

        modes = []
        for mode in self._loss_modes:
            if len(mode.value) > 2:
                for k in mode.value[2]:
                    modes.append(LossModes[k])
            modes.append(mode)
        # TODO: resolve requirements chain and put modes in correct order, not just 'hopefully correct' one
        self._losses_to_calculate = list(dict.fromkeys(modes))

        # default metrics
        self.metrics = MetricCollection([
            MPJPE(dist_sync_on_step=True),
            MRPE(
                dist_sync_on_step=True,
                input_nodes=self.movements_model.input_nodes,
                output_nodes=self.movements_model.output_nodes
            ),
            FB_MPJPE(dist_sync_on_step=True),
            # FB_WeightedMPJPE should be same as FB_MPJPE since we provide no weights:
            FB_WeightedMPJPE(dist_sync_on_step=True),
            FB_PA_MPJPE(dist_sync_on_step=True),
            FB_N_MPJPE(dist_sync_on_step=True),
            FB_MPJVE(dist_sync_on_step=True),
        ])

        self.save_hyperparameters({
            'host': platform.node(),
            'loss_modes': [mode.name for mode in self._loss_modes],
            **self.movements_model.hparams,
            **self.trajectory_model.hparams,
        })

    def configure_optimizers(self):
        movements_optimizers = self.movements_model.configure_optimizers()
        trajectory_optimizers = self.trajectory_model.configure_optimizers()

        if 'optimizer' not in movements_optimizers:
            movements_optimizers = None

        if 'optimizer' not in trajectory_optimizers:
            trajectory_optimizers = None

        return [opt for opt in [movements_optimizers, trajectory_optimizers] if opt is not None]

    @ staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific commandline arguments.

        By default, this method adds parameters for projection layer and loss modes.
        If overriding, remember to call super().
        """

        parser = parent_parser.add_argument_group("BaseMapper Module")
        parser.add_argument(
            '--input_nodes',
            type=get_skeleton_type_by_name,
            default=CARLA_SKELETON
        )
        parser.add_argument(
            '--output_nodes',
            type=get_skeleton_type_by_name,
            default=CARLA_SKELETON
        )
        parser.add_argument(
            '--loss_modes',
            help="""
                Set loss modes to use in the preferred order.
                Choices: {}.
                Default: ['common_loc_2d']
                """.format(
                set(LossModes.__members__.keys())),
            metavar="MODE",
            default=[],
            choices=list(LossModes),
            nargs="+",
            action="extend",
            type=LossModes.__getitem__
        )

        return parent_parser

    @rank_zero_only
    def on_train_start(self):
        # We need to manually add the datamodule hparams,
        # because the merge is automatically handled only for initial_hparams
        # in the Trainer.
        hparams = self.hparams
        hparams.update(self.trainer.datamodule.hparams)

        self.logger[0].log_hyperparams(hparams, {
            "hp/{}".format(k): 0
            for k in self.metrics.keys()
        })

    def _on_batch_start(self, batch, batch_idx):
        self.projection.on_batch_start(batch, batch_idx)

    def on_train_batch_start(self, batch, batch_idx, *args, **kwargs):
        self._on_batch_start(batch, batch_idx)

    def on_validation_batch_start(self, batch, batch_idx, *args, **kwargs):
        self._on_batch_start(batch, batch_idx)

    def on_test_batch_start(self, batch, batch_idx, *args, **kwargs):
        self._on_batch_start(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        self._on_batch_start(batch, batch_idx)
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'test')

    def validation_step_end(self, outputs):
        self._eval_step_end(outputs, 'val')

    def test_step_end(self, outputs):
        self._eval_step_end(outputs, 'test')

    def training_epoch_end(self, outputs: Any) -> None:
        to_log = {}

        if hasattr(self.movements_model, 'training_epoch_end'):
            to_log.update(self.movements_model.training_epoch_end(outputs))

        if hasattr(self.trajectory_model, 'training_epoch_end'):
            to_log.update(self.trajectory_model.training_epoch_end(outputs))

        if len(to_log) > 0:
            batch_size = len(outputs[0]['preds']['absolute_pose_loc'])
            self.log_dict(to_log, batch_size=batch_size)

    def _step(self, batch, batch_idx, stage):
        (frames, targets, meta) = batch

        no_conf_frames = frames[..., 0:2].clone()

        pose_inputs = self.movements_model(
            frames if self.movements_model.needs_confidence else no_conf_frames,
            targets if self.training else None
        )

        world_loc_inputs, world_rot_inputs = self.trajectory_model(
            no_conf_frames,
            targets if self.training else None
        )

        (projected_pose, normalized_projection,
         absolute_pose_loc, absolute_pose_rot,
         world_loc, world_rot) = self.projection(
            pose_inputs,
            world_loc_inputs,
            world_rot_inputs
        )

        # TODO: this will work for mono-type batches, but not for mixed-type batches;
        # Figure if/how to do mixed-type batches - should we even support it?
        # Maybe force reduction='none' in criterions and then reduce here?
        loss_dict = {}
        # TODO: this should take into account both movements and trajectory models
        eval_slice = (slice(None), self.movements_model.eval_slice)

        # get all inputs/outputs properly sliced
        ev_pose_changes = tuple([v[eval_slice] for v in pose_inputs]) if isinstance(
            pose_inputs, tuple) else pose_inputs[eval_slice]
        ev_projected_pose = projected_pose[eval_slice]
        ev_normalized_projection = normalized_projection[eval_slice]
        ev_absolute_pose_loc = absolute_pose_loc[eval_slice]
        ev_absolute_pose_rot = absolute_pose_rot[eval_slice] if absolute_pose_rot is not None else None
        ev_world_loc_inputs = world_loc_inputs[eval_slice]
        ev_world_rot_inputs = world_rot_inputs[eval_slice] if world_rot_inputs is not None else None
        ev_world_loc = world_loc[eval_slice]
        ev_world_rot = world_rot[eval_slice] if world_rot is not None else None
        ev_frames = frames[eval_slice]
        ev_targets = {k: v[eval_slice] for k, v in targets.items()}

        for mode in self._losses_to_calculate:
            (loss_fn, criterion, *_) = mode.value
            loss = loss_fn(
                criterion=criterion,
                input_nodes=self.movements_model.input_nodes,
                pose_changes=ev_pose_changes,
                projected_pose=ev_projected_pose,
                normalized_projection=ev_normalized_projection,
                absolute_pose_loc=ev_absolute_pose_loc,
                absolute_pose_rot=ev_absolute_pose_rot,
                world_loc=ev_world_loc,
                world_rot=ev_world_rot,
                frames=ev_frames,
                targets=ev_targets,
                meta=meta,
                requirements={
                    k.name: v
                    for k, v in loss_dict.items()
                    if k.name in mode.value[2]
                } if len(mode.value) > 2 else None,
            )
            if loss is not None and not torch.isnan(loss):
                loss_dict[mode] = loss

        for k, v in loss_dict.items():
            self.log('{}_loss/{}'.format(stage, k.name), v, batch_size=len(frames))

        self._log_videos(
            projected_pose=ev_projected_pose,
            absolute_pose_loc=ev_absolute_pose_loc,
            absolute_pose_rot=ev_absolute_pose_rot,
            world_loc=ev_world_loc,
            world_rot=ev_world_rot,
            inputs=ev_frames,
            targets=ev_targets,
            meta=meta,
            batch_idx=batch_idx,
            stage=stage
        )

        # return primary loss - the first one available from loss_modes list
        # also log it as 'primary' for monitoring purposes
        # TODO: monitoring should be done based on the metric, not the loss
        # so 'primary' loss should be removed in the future
        for mode in self._loss_modes:
            if mode in loss_dict:
                self.log('{}_loss/primary'.format(stage),
                         loss_dict[mode], batch_size=len(frames))
                return {
                    'loss': loss_dict[mode],
                    'preds': {
                        'pose_changes': ev_pose_changes.detach() if self.movements_model.output_type == MovementsModelOutputType.pose_changes else None,
                        'world_rot_changes': ev_world_rot_inputs.detach() if self.trajectory_model.output_type == TrajectoryModelOutputType.changes else None,
                        'world_loc_changes': ev_world_loc_inputs.detach() if self.trajectory_model.output_type == TrajectoryModelOutputType.changes else None,
                        'absolute_pose_loc': ev_absolute_pose_loc.detach(),
                        'absolute_pose_rot': ev_absolute_pose_rot.detach() if ev_absolute_pose_rot is not None else None,
                        'world_loc': ev_world_loc.detach(),
                        'world_rot': ev_world_rot.detach(),
                    },
                    'targets': ev_targets
                }

        raise RuntimeError("Couldn't calculate any loss.")

    def _eval_step_end(self, outputs, stage):
        # calculate and log metrics
        m = self.metrics(outputs['preds'], outputs['targets'])
        batch_size = len(outputs['preds']['absolute_pose_loc'])
        for k, v in m.items():
            self.log('hp/{}'.format(k), v,
                     batch_size=batch_size)

    def _log_to_tensorboard(self, vid, vid_idx, fps, stage, meta):
        vid = vid.permute(0, 1, 4, 2, 3).unsqueeze(0)  # B,T,H,W,C -> B,T,C,H,W
        self.logger[0].experiment.add_video(
            '{}_{}_render'.format(stage, vid_idx),
            vid, self.global_step, fps=fps
        )

    def _log_videos(self,
                    projected_pose: Tensor,
                    absolute_pose_loc: Tensor,
                    absolute_pose_rot: Tensor,
                    world_loc: Tensor,
                    world_rot: Tensor,
                    inputs: Tensor,
                    targets: Tensor,
                    meta: Tensor,
                    batch_idx: int,
                    stage: str,
                    log_to_tb: bool = False
                    ):
        if log_to_tb:
            vid_callback = self._log_to_tensorboard
        else:
            vid_callback = None

        self.logger[1].experiment.log_videos(
            inputs,
            targets,
            meta,
            projected_pose,
            absolute_pose_loc,
            absolute_pose_rot,
            world_loc,
            world_rot,
            self.global_step,
            batch_idx,
            stage,
            vid_callback,
            force=(stage != 'train')
        )
