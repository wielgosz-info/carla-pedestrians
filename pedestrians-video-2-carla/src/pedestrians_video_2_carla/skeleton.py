import argparse
import logging
import sys
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from pedestrians_video_2_carla import __version__
from pedestrians_video_2_carla.loggers.pedestrian_logger import PedestrianLogger
from pedestrians_video_2_carla.data.datamodules import *
from pedestrians_video_2_carla.modules.lightning import *

__author__ = "Maciej Wielgosz"
__copyright__ = "Maciej Wielgosz"
__license__ = "MIT"

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


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
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.INFO)


def main(args):
    """

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose"]``).
    """

    parser = add_program_args()
    parser = pl.Trainer.add_argparse_args(parser)

    parser = Carla2D3DDataModule.add_data_specific_args(parser)
    parser = LitLSTMMapper.add_model_specific_args(parser)

    args = parser.parse_args(args)
    setup_logging(args.loglevel)

    dict_args = vars(args)

    # data
    dm = Carla2D3DDataModule(**dict_args)

    # if model needs to know something about the data:
    # dm.prepare_data()
    # dm.setup()

    # model
    model = LitLSTMMapper(**dict_args,
                          log_videos_every_n_epochs=21,
                          enabled_renderers={
                              'source': False,
                              'input': True,
                              'projection': True,
                              'carla': False
                          })

    # loggers - use TensorBoardLogger log dir as default for all loggers & checkpoints
    tb_logger = TensorBoardLogger(
        'lightning_logs',
        name=model.__class__.__name__
    )
    pedestrian_logger = PedestrianLogger(
        save_dir=os.path.join(tb_logger.log_dir, 'videos'),
        name=tb_logger.name,
        version=tb_logger.version
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
        monitor="val_loss",
        save_top_k=1
    )

    # training
    trainer = pl.Trainer.from_argparse_args(args, logger=[
        tb_logger,
        pedestrian_logger,
    ], callbacks=[
        checkpoint_callback
    ])
    trainer.fit(model=model, datamodule=dm)

    # testing
    # trainer.test(model=model, datamodule=dm,
    #              ckpt_path='/app/lightning_logs/version_2/checkpoints/epoch=209-step=6719.ckpt')


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m pedestrians_video_2_carla.skeleton 42
    #
    run()
