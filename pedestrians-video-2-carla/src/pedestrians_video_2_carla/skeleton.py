import argparse
import logging
import sys
import os
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from pedestrians_video_2_carla import __version__
from pedestrians_video_2_carla.loggers.pedestrian_logger import PedestrianLogger
from pedestrians_video_2_carla.data.datamodules import *
from pedestrians_video_2_carla.modules.lightning import *
from pedestrians_video_2_carla.transforms.hips_neck import CarlaHipsNeckExtractor, HipsNeckNormalize

__author__ = "Maciej Wielgosz"
__copyright__ = "Maciej Wielgosz"
__license__ = "MIT"

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


# TODO: get this from argparse
def get_model_cls():
    return LitLinearAutoencoderMapper


# TODO: get this from argparse
def get_data_module_cls():
    return Carla2D3DDataModule


# TODO: this probably should be encapsulated in DataModule
# if you need different transforms, you need different data modules
def get_data_transform(nodes):
    return HipsNeckNormalize(CarlaHipsNeckExtractor(nodes))


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

    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.INFO)


def main(args: List[str]):
    """
    :param args: command line parameters as list of strings
          (for example  ``["--verbose"]``).
    :type args: List[str]
    """

    model_cls = get_model_cls()
    data_module_cls = get_data_module_cls()

    parser = add_program_args()
    parser = pl.Trainer.add_argparse_args(parser)

    parser = data_module_cls.add_data_specific_args(parser)
    parser = model_cls.add_model_specific_args(parser)
    parser = PedestrianLogger.add_logger_specific_args(parser)

    args = parser.parse_args(args)
    setup_logging(args.loglevel)

    dict_args = vars(args)

    # data
    dm = data_module_cls(
        transform=get_data_transform,
        **dict_args
    )

    # model
    model = model_cls(**dict_args)

    # loggers - use TensorBoardLogger log dir as default for all loggers & checkpoints
    tb_logger = TensorBoardLogger(
        'lightning_logs',
        name=model.__class__.__name__
    )
    pedestrian_logger = PedestrianLogger(
        save_dir=os.path.join(tb_logger.log_dir, 'videos'),
        name=tb_logger.name,
        version=tb_logger.version,
        **dict_args
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    # training
    trainer = pl.Trainer.from_argparse_args(args, logger=[
        tb_logger,
        pedestrian_logger,
    ], callbacks=[
        checkpoint_callback
    ])

    if args.mode == 'train':
        trainer.fit(model=model, datamodule=dm)
    elif args.mode == 'test':
        trainer.test(model=model, datamodule=dm)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
