
from typing import List, Tuple, Type

import pytorch_lightning as pl
from pedestrians_video_2_carla.metrics.mpjpe import MPJPE
from pedestrians_video_2_carla.modules.loss import LossModes
from pedestrians_video_2_carla.modules.projection.projection import \
    ProjectionModule, ProjectionTypes
from pedestrians_video_2_carla.skeletons.nodes import (
    Skeleton, get_skeleton_name_by_type, get_skeleton_type_by_name)
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from torch.functional import Tensor
import platform
from torchmetrics import MetricCollection


class LitBaseMapper(pl.LightningModule):
    """
    Base LightningModule - all other LightningModules should inherit from this.
    It contains projection layer, loss modes handling and video logging.

    Derived LightningModules should implement forward() and configure_optimizers() method.
    If they use additional hyperparameters, they should also call self.save_hyperparameters({...})
    in __init__() and (optionally) override add_model_specific_args() method.
    """

    def __init__(
        self,
        input_nodes: Type[Skeleton] = CARLA_SKELETON,
        output_nodes: Type[Skeleton] = CARLA_SKELETON,
        loss_modes: List[LossModes] = None,
        projection_type: ProjectionTypes = ProjectionTypes.pose_changes,
        needs_confidence: bool = False,
        **kwargs
    ):
        super().__init__()

        if loss_modes is None or len(loss_modes) == 0:
            loss_modes = [LossModes.common_loc_2d]
        self._loss_modes = loss_modes

        self._needs_confidence = needs_confidence

        modes = []
        for mode in self._loss_modes:
            if len(mode.value) > 2:
                for k in mode.value[2]:
                    modes.append(LossModes[k])
            modes.append(mode)
        # TODO: resolve requirements chain and put modes in correct order, not just 'hopefully correct' one

        self._losses_to_calculate = list(dict.fromkeys(modes))

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        # default layer
        self.projection = ProjectionModule(
            projection_type=projection_type,
            **kwargs
        )

        # default metrics
        self.metrics = MetricCollection([
            MPJPE(dist_sync_on_step=True),
        ])

        self.save_hyperparameters({
            'input_nodes': get_skeleton_name_by_type(self.input_nodes),
            'output_nodes': get_skeleton_name_by_type(self.output_nodes),
            'loss_modes': [mode.name for mode in self._loss_modes],
            'projection_type': projection_type.name,
            'model_name': self.__class__.__name__,
            'host': platform.node(),
        })

    @ staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific commandline arguments.

        By default, this method adds parameters for projection layer and loss modes.
        If overriding, remember to call super().
        """

        parser = parent_parser.add_argument_group("BaseMapper Lightning Module")
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

    def on_train_start(self):
        self.logger[0].log_hyperparams(self.hparams, {
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

    def _step(self, batch, batch_idx, stage):
        (frames, targets, meta) = batch

        if self._needs_confidence:
            frames = frames.to(self.device)
        else:
            frames = frames[..., 0:2].clone().to(self.device)

        pose_inputs = self.forward(frames, targets if self.training else None)

        (projected_pose, normalized_projection, absolute_pose_loc, absolute_pose_rot) = self.projection(
            pose_inputs
        )

        # TODO: this will work for mono-type batches, but not for mixed-type batches;
        # Figure if/how to do mixed-type batches - should we even support it?
        # Maybe force reduction='none' in criterions and then reduce here?
        loss_dict = {}

        for mode in self._losses_to_calculate:
            (loss_fn, criterion, *_) = mode.value
            loss = loss_fn(
                criterion=criterion,
                input_nodes=self.input_nodes,
                pose_changes=pose_inputs,
                projected_pose=projected_pose,
                normalized_projection=normalized_projection,
                absolute_pose_loc=absolute_pose_loc,
                absolute_pose_rot=absolute_pose_rot,
                frames=frames,
                targets=targets,
                meta=meta,
                requirements={
                    k.name: v
                    for k, v in loss_dict.items()
                    if k.name in mode.value[2]
                } if len(mode.value) > 2 else None
            )
            if loss is not None:
                loss_dict[mode] = loss

        for k, v in loss_dict.items():
            self.log('{}_loss/{}'.format(stage, k.name), v, batch_size=len(frames))

        self._log_videos(
            projected_pose=projected_pose,
            absolute_pose_loc=absolute_pose_loc,
            absolute_pose_rot=absolute_pose_rot,
            batch=batch,
            batch_idx=batch_idx,
            stage=stage
        )

        # return primary loss - the first one available from loss_modes list
        # also log it as 'primary' for monitoring purposes
        # TODO: monitoring should be done based on the metric, not the loss
        # so 'primary' loss should be removed in the future
        for mode in self._loss_modes:
            if mode in loss_dict:
                self.log('{}_loss/primary'.format(stage), loss_dict[mode])
                return {
                    'loss': loss_dict[mode],
                    'preds': {
                        'pose_changes': pose_inputs.detach(),
                        'world_rot_changes': None,  # TODO: implement this; should those be changes or abs?
                        'world_loc_changes': None,  # TODO: implement this; should those be changes or abs?
                        'absolute_pose_loc': absolute_pose_loc.detach(),
                        'absolute_pose_rot': absolute_pose_rot.detach(),
                    },
                    'targets': targets
                }

        raise RuntimeError("Couldn't calculate any loss.")

    def _eval_step_end(self, outputs, stage):
        # calculate and log metrics
        m = self.metrics(outputs['preds'], outputs['targets'])
        for k, v in m.items():
            self.log('hp/{}'.format(k), v)

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
                    batch: Tuple,
                    batch_idx: int,
                    stage: str,
                    log_to_tb: bool = False
                    ):
        if log_to_tb:
            vid_callback = self._log_to_tensorboard
        else:
            vid_callback = None

        self.logger[1].experiment.log_videos(
            batch,
            projected_pose,
            absolute_pose_loc,
            absolute_pose_rot,
            self.global_step,
            batch_idx,
            stage,
            vid_callback,
            force=(stage != 'train')
        )
