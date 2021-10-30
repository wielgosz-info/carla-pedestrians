from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment


class PedestrianLogger(LightningLoggerBase):
    """
    Logger for video output.
    """

    def __init__(self, save_dir, log_every_n_epochs=10, fps=30.0, max_videos=10):
        """
        Initialize PedestrianLogger.

        Args:
            save_dir (str): directory to save videos
            log_every_n_epochs (int): interval in epochs to save videos
            fps (int): frames per second of video
            max_videos (int): maximum number of videos from the single batch to save
        """
        self.save_dir = save_dir
        self.log_every_n_epochs = log_every_n_epochs
        self.fps = fps
        self.max_videos = max_videos

    @property
    def name(self):
        return "PedestrianLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
