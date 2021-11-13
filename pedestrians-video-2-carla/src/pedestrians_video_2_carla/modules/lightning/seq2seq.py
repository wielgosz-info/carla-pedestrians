from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.functional import Tensor

from pedestrians_video_2_carla.modules.lightning.base import LitBaseMapper


class Encoder(nn.Module):
    def __init__(self, hid_dim=64, n_layers=2, dropout=0.2, input_nodes_len=26):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.__input_nodes_len = input_nodes_len
        self.__input_features = 2  # (x, y) points
        self.__input_size = self.__input_nodes_len * self.__input_features

        self.rnn = nn.LSTM(self.__input_size, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        original_shape = x.shape
        x = x.view(*original_shape[0:2], self.__input_size)
        _, (hidden, cell) = self.rnn(x)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, hid_dim=64, n_layers=2, dropout=0.2, output_nodes_len=26):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.__output_nodes_len = output_nodes_len
        self.__output_features = 6  # Rotation 6D
        self.output_size = self.__output_nodes_len * self.__output_features

        self.rnn = nn.LSTM(self.output_size, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, self.output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        output, (hidden, cell) = self.rnn(x, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(LitBaseMapper):
    """
    Sequence to sequence model
    """

    def __init__(self, hidden_size=64, num_layers=2, p_dropout=0.2, **kwargs):
        super().__init__(**kwargs)

        self.encoder = Encoder(
            hid_dim=hidden_size, n_layers=num_layers, dropout=p_dropout,
            input_nodes_len=len(self.input_nodes)
        )
        self.decoder = Decoder(
            hid_dim=hidden_size, n_layers=num_layers, dropout=p_dropout,
            output_nodes_len=len(self.output_nodes)
        )

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        self.save_hyperparameters({
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'p_dropout': p_dropout
        })

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = LitBaseMapper.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("Seq2Seq Lightning Module")
        parser.add_argument(
            '--num_layers',
            default=2,
            type=int,
        )
        parser.add_argument(
            '--hidden_size',
            default=64,
            type=int,
        )
        parser.add_argument(
            '--p_dropout',
            default=0.2,
            type=float,
        )
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape

        # convert to sequence-first format
        x = x.permute(1, 0, *range(2, x.dim()))

        batch_size = original_shape[0]
        clip_length = original_shape[1]

        # tensor to store decoder outputs
        outputs = torch.zeros(
            (clip_length, batch_size, self.decoder.output_size), device=x.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x)

        # first input to the decoder is the <sos> tokens
        input = torch.zeros((batch_size, self.decoder.output_size), device=x.device)

        for t in range(0, clip_length):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output
            input = output

        # convert back to batch-first format
        outputs = outputs.permute(1, 0, 2)

        return rotation_6d_to_matrix(outputs.view(*original_shape[:3], 6))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)

        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, cooldown=10),
            'interval': 'epoch',
            'monitor': 'train_loss/primary'
        }

        config = {
            'lr_scheduler': lr_scheduler,
            'optimizer': optimizer
        }

        return config
