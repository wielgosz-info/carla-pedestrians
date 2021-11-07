
from typing import List, Tuple, Type

import pytorch_lightning as pl
from pedestrians_video_2_carla.modules.loss import LossModes
from pedestrians_video_2_carla.modules.projection.projection import \
    ProjectionModule, ProjectionTypes
from pedestrians_video_2_carla.skeletons.nodes import (
    Skeleton, get_skeleton_name_by_type, get_skeleton_type_by_name)
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.openpose import BODY_25_SKELETON
from torch.functional import Tensor


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
        input_nodes: Type[Skeleton] = BODY_25_SKELETON,
        output_nodes: Type[Skeleton] = CARLA_SKELETON,
        loss_modes: List[LossModes] = None,
        projection_type: ProjectionTypes = ProjectionTypes.pose_changes,
        **kwargs
    ):
        super().__init__()

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

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        # default layer
        self.projection = ProjectionModule(
            projection_type=projection_type,
            **kwargs
        )

        self.save_hyperparameters({
            'input_nodes': get_skeleton_name_by_type(self.input_nodes),
            'output_nodes': get_skeleton_name_by_type(self.output_nodes),
            'loss_modes': [mode.name for mode in self._loss_modes],
            'projection_type': projection_type.name
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
            default=BODY_25_SKELETON
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

    def _on_batch_start(self, batch, batch_idx, dataloader_idx):
        self.projection.on_batch_start(batch, batch_idx, dataloader_idx)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, None, 'train')

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._step(batch, batch_idx, dataloader_idx, 'test')

    def _step(self, batch, batch_idx, dataloader_idx, stage):
        (frames, targets, meta) = batch

        pose_changes = self.forward(frames.to(self.device))

        (projected_pose, normalized_projection, absolute_pose_loc, absolute_pose_rot) = self.projection(
            pose_changes
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
                pose_changes=pose_changes,
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
            self.log('{}_loss/{}'.format(stage, k.name), v)

        self._log_videos(pose_changes, projected_pose, batch,
                         batch_idx, dataloader_idx, stage)

        # return primary loss - the first one available from loss_modes list
        # also log it as 'primary' for monitoring purposes
        # TODO: monitoring should be done based on the metric, not the loss
        # so 'primary' loss should be removed in the future
        for mode in self._loss_modes:
            if mode in loss_dict:
                self.log('{}_loss/primary'.format(stage), loss_dict[mode])
                return loss_dict[mode]

        raise RuntimeError("Couldn't calculate any loss.")

    def _log_to_tensorboard(self, vid, vid_idx, fps, stage, meta):
        vid = vid.permute(0, 1, 4, 2, 3).unsqueeze(0)  # B,T,H,W,C -> B,T,C,H,W
        self.logger[0].experiment.add_video(
            '{}_{}_render'.format(stage, vid_idx),
            vid, self.global_step, fps=fps
        )

    def _log_videos(self, pose_change: Tensor, projected_pose: Tensor, batch: Tuple, batch_idx: int, dataloader_idx: int, stage: str, log_to_tb: bool = False):
        if log_to_tb:
            vid_callback = self._log_to_tensorboard
        else:
            vid_callback = None

        self.logger[1].experiment.log_videos(
            batch, projected_pose, pose_change, self.global_step, batch_idx, dataloader_idx, stage, vid_callback, force=(stage != 'train'))
