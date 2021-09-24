
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class LitBaseMapper(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def _on_batch_start(self, batch, batch_idx, dataloader_idx):
        pass

    def _pose_projection(self, pose_change):
        pass

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        return self(batch)
