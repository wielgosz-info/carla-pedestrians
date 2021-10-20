import argparse
import logging
import sys

import pytorch_lightning as pl

from pedestrians_video_2_carla import __version__
from pedestrians_video_2_carla.pl_datamodules.jaad_openpose import JAADOpenPoseDataModule
from pedestrians_video_2_carla.pl_modules.linear import LitLinearMapper

__author__ = "Maciej Wielgosz"
__copyright__ = "Maciej Wielgosz"
__license__ = "MIT"

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Map pedestrians movements from videos to CARLA")
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
    return parser.parse_args(args)


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
    args = parse_args(args)
    setup_logging(args.loglevel)

    # data
    dm = JAADOpenPoseDataModule()

    # if model needs to know something about the data:
    # openpose_dm.prepare_data()
    # openpose_dm.setup()

    # model
    model = LitLinearMapper()

    # training
    trainer = pl.Trainer(gpus=1, log_every_n_steps=1, max_epochs=2000)
    trainer.fit(model, datamodule=dm)

    # testing
    trainer.test(datamodule=dm)


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
