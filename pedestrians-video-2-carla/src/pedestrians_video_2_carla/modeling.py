import argparse
import logging
import sys
import os
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from pedestrians_video_2_carla import __version__
from pedestrians_video_2_carla.data.datamodules.base import BaseDataModule
from pedestrians_video_2_carla.loggers.pedestrian import PedestrianLogger
from pedestrians_video_2_carla.data.datamodules import DATA_MODULES
from pedestrians_video_2_carla.modules.base.base import LitBaseMapper
from pedestrians_video_2_carla.modules.base.movements import MovementsModel
from pedestrians_video_2_carla.modules.base.trajectory import TrajectoryModel
from pedestrians_video_2_carla.modules.movements import MOVEMENTS_MODELS
from pedestrians_video_2_carla.modules.trajectory import TRAJECTORY_MODELS

__author__ = "Maciej Wielgosz"
__copyright__ = "Maciej Wielgosz"
__license__ = "MIT"

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def get_base_model_cls() -> LitBaseMapper:
    return LitBaseMapper


def get_movements_model_cls(model_name: str = "Baseline3DPoseRot") -> MovementsModel:
    return MOVEMENTS_MODELS[model_name]


def get_trajectory_model_cls(model_name: str = "ZeroTrajectory") -> TrajectoryModel:
    return TRAJECTORY_MODELS[model_name]


def get_data_module_cls(data_module_name: str = "Carla2D3D") -> BaseDataModule:
    return DATA_MODULES[data_module_name]


def add_program_args():
    """
    Add program-level command line parameters
    """
    parser = argparse.ArgumentParser(
        description="Map pedestrians movements from videos to CARLA"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="pedestrians-video-2-carla {ver}".format(ver=__version__),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very_verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    parser.add_argument(
        "-m",
        "--mode",
        dest="mode",
        help="set mode to train or test",
        default="train",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--data_module_name",
        dest="data_module_name",
        help="Data module class to use",
        default="Carla2D3D",
        choices=list(DATA_MODULES.keys()),
        type=str,
    )
    parser.add_argument(
        "--movements_model_name",
        dest="movements_model_name",
        help="Movements model class to use",
        default="Baseline3DPoseRot",
        choices=list(MOVEMENTS_MODELS.keys()),
        type=str,
    )
    parser.add_argument(
        "--trajectory_model_name",
        dest="trajectory_model_name",
        help="Trajectory model class to use",
        default="ZeroTrajectory",
        choices=list(TRAJECTORY_MODELS.keys()),
        type=str,
    )
    parser.add_argument(
        "--logs_dir",
        dest="logs_dir",
        default=os.path.join(os.getcwd(), "lightning_logs"),
        type=str,
    )
    return parser


def setup_logging(loglevel):
    """
    Setup basic logging

    :param loglevel: minimum loglevel for emitting messages
    :type loglevel: int
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.INFO)


def main(args: List[str]):
    """
    :param args: command line parameters as list of strings
          (for example  ``["--verbose"]``).
    :type args: List[str]
    """

    parser = add_program_args()
    tmp_args = args[:]
    try:
        tmp_args.remove("-h")
    except ValueError:
        pass
    try:
        tmp_args.remove("--help")
    except ValueError:
        pass
    program_args, _ = parser.parse_known_args(tmp_args)

    parser = pl.Trainer.add_argparse_args(parser)

    data_module_cls = get_data_module_cls(program_args.data_module_name)
    base_model_cls = get_base_model_cls()
    movements_model_cls = get_movements_model_cls(program_args.movements_model_name)
    trajectory_model_cls = get_trajectory_model_cls(program_args.trajectory_model_name)

    parser = data_module_cls.add_data_specific_args(parser)
    parser = base_model_cls.add_model_specific_args(parser)
    parser = movements_model_cls.add_model_specific_args(parser)
    parser = trajectory_model_cls.add_model_specific_args(parser)

    parser = PedestrianLogger.add_logger_specific_args(parser)

    args = parser.parse_args(args)
    setup_logging(args.loglevel)

    dict_args = vars(args)

    # data
    dm = data_module_cls(**dict_args)

    # model
    movements_model = movements_model_cls(**dict_args)
    trajectory_model = trajectory_model_cls(**dict_args)
    model = base_model_cls(
        movements_model=movements_model,
        trajectory_model=trajectory_model,
        **dict_args
    )

    # loggers - use TensorBoardLogger log dir as default for all loggers & checkpoints
    tb_logger = TensorBoardLogger(
        args.logs_dir,
        name=os.path.join(
            trajectory_model.__class__.__name__,
            movements_model.__class__.__name__
        ),
        default_hp_metric=False
    )

    # some models support this as a CLI option
    # so we only add it if it's not already set
    dict_args.setdefault("movements_output_type",
                         movements_model.output_type)

    pedestrian_logger = PedestrianLogger(
        save_dir=os.path.join(tb_logger.log_dir, "videos"),
        name=tb_logger.name,
        version=tb_logger.version,
        **dict_args
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        monitor="val_loss/primary",
        mode="min",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # training
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=[tb_logger, pedestrian_logger, ],
        callbacks=[checkpoint_callback, lr_monitor],
    )

    if args.mode == "train":
        trainer.fit(model=model, datamodule=dm)
    elif args.mode == "test":
        trainer.test(model=model, datamodule=dm)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
