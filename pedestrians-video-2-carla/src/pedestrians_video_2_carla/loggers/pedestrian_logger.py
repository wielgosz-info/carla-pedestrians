from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
import os


class PedestrianWriter(object):
    def __init__(self, log_dir, log_every_n_epochs=10, fps=30.0, max_videos=10, **kwargs) -> None:
        self._log_dir = log_dir

        self._log_every_n_epochs = log_every_n_epochs
        self._fps = fps
        self._max_videos = max_videos


class PedestrianLogger(LightningLoggerBase):
    """
    Logger for video output.
    """

    def __init__(self, save_dir, name, version, **kwargs):
        """
        Initialize PedestrianLogger.

        Args:
            save_dir (str): directory to save videos. Usually you get this from TensorBoardLogger.log_dir + 'videos'.
            log_every_n_epochs (int): interval in epochs to save videos
            fps (int): frames per second of video
            max_videos (int): maximum number of videos from the single batch to save
        """
        super().__init__(**kwargs)

        self._save_dir = save_dir
        self._name = name
        self._version = version
        self._kwargs = kwargs

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        return self._version

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            self._experiment = PedestrianWriter(log_dir=self._save_dir, **self._kwargs)

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        # we do not log any hyperparams
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        # if aggregating, this will be called every n steps and whenever checkpoint is saved
        pass
