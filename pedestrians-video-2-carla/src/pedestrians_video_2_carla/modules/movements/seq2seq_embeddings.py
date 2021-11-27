from typing import Dict
from pytorch3d.transforms.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.functional import Tensor
from pedestrians_video_2_carla.modules.base.movements import MovementsModel
from .seq2seq import TeacherMode, Encoder, Decoder


class Seq2SeqEmbeddings(MovementsModel):
    """
    Sequence to sequence model
    """

    def __init__(self,
                 hidden_size=64,
                 num_layers=2,
                 p_dropout=0.2,
                 single_joint_embeddings_size=64,
                 teacher_mode: TeacherMode = TeacherMode.no_force,
                 teacher_force_ratio: float = 0.2,
                 teacher_force_drop: float = 0.02,
                 **kwargs):
        super().__init__(**kwargs)

        self.teacher_mode = teacher_mode
        self.teacher_force_ratio = teacher_force_ratio
        self.teacher_force_drop = teacher_force_drop
        self.single_joint_embeddings_size = single_joint_embeddings_size

        self.embeddings = nn.ModuleList([nn.Linear(2, self.single_joint_embeddings_size)
                                         for _ in range(len(self.input_nodes))])
        self.encoder = Encoder(
            hid_dim=hidden_size, n_layers=num_layers, dropout=p_dropout,
            input_nodes_len=len(self.input_nodes),
            input_features=self.single_joint_embeddings_size
        )
        self.decoder = Decoder(
            hid_dim=hidden_size, n_layers=num_layers, dropout=p_dropout,
            output_nodes_len=len(self.output_nodes)
        )

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        self._hparams = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'p_dropout': p_dropout,
            'teacher_mode': self.teacher_mode.name,
            'teacher_force_ratio': self.teacher_force_ratio,
            'teacher_force_drop': self.teacher_force_drop,
            'single_joint_embeddings_size': self.single_joint_embeddings_size
        }

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Seq2SeqEmbeddings Movements Module")
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
        parser.add_argument(
            '--teacher_mode',
            help="""
                Set teacher mode for decoder training.
                """.format(
                set(TeacherMode.__members__.keys())),
            default=TeacherMode.no_force,
            choices=list(TeacherMode),
            type=TeacherMode.__getitem__
        )
        parser.add_argument(
            '--teacher_force_ratio',
            help="""
                Set teacher force ratio for decoder training.
                Only used if teacher_mode is not TeacherMode.no_force.
                """,
            default=0.2,
            type=float
        )
        parser.add_argument(
            '--teacher_force_drop',
            help="""
                Set teacher force ratio drop per epoch for decoder training.
                Only used if teacher_mode is not TeacherMode.no_force.
                """,
            default=0.02,
            type=float
        )
        parser.add_argument(
            '--single_joint_embeddings_size',
            default=64,
            type=int,
        )

        return parent_parser

    def forward(self, x: Tensor, targets: Dict[str, Tensor] = None, *args, **kwargs) -> Tensor:
        original_shape = x.shape

        # convert to sequence-first format
        x = x.permute(1, 0, *range(2, x.dim()))

        batch_size = original_shape[0]
        clip_length = original_shape[1]
        joints = original_shape[2]

        assert joints == len(self.input_nodes)
        assert joints == len(self.embeddings)

        # tensore to store the embeddings
        embeddings = torch.zeros(
            (clip_length, batch_size, joints, self.single_joint_embeddings_size),
            dtype=torch.float32,
            device=x.device
        )

        # get embeddings
        for i, embedding in enumerate(self.embeddings):
            embeddings[:, :, i, :] = embedding(x[:, :, i, :])

        # tensor to store decoder outputs
        outputs = torch.zeros(
            (clip_length, batch_size, self.decoder.output_size), device=x.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(embeddings)

        # first input to the decoder is the <sos> tokens
        input = torch.zeros((batch_size, self.decoder.output_size), device=x.device)

        needs_forcing, target_pose_changes, force_indices = self.__teacher_forcing(
            targets)

        for t in range(0, clip_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output
            input = output

            if needs_forcing:
                input[force_indices[t]] = target_pose_changes[t, force_indices[t]]

        # convert back to batch-first format
        outputs = outputs.permute(1, 0, 2)

        return rotation_6d_to_matrix(outputs.view(*original_shape[:3], 6))

    def __teacher_forcing(self, targets):
        needs_forcing = self.training and self.teacher_mode != TeacherMode.no_force and targets is not None and self.teacher_force_ratio > 0
        target_pose_changes = None
        force_indices = None

        if needs_forcing:
            target_pose_changes = matrix_to_rotation_6d(targets['pose_changes'])

            (batch_size, clip_length, *_) = target_pose_changes.shape

            target_pose_changes = target_pose_changes.permute(
                1, 0, *range(2, target_pose_changes.dim())).reshape((clip_length, batch_size, self.decoder.output_size))

            if self.teacher_mode == TeacherMode.clip_force:
                # randomly select clips that should be taken from targets
                force_indices = (torch.rand(
                    (1, batch_size), device=target_pose_changes.device) < self.teacher_force_ratio).repeat((clip_length, 1))
            elif self.teacher_mode == TeacherMode.frames_force:
                # randomly select frames that should be taken from targets
                force_indices = torch.rand(
                    (clip_length, batch_size), device=target_pose_changes.device) < self.teacher_force_ratio

        return needs_forcing, target_pose_changes, force_indices

    def training_epoch_end(self, *args, **kwargs) -> Dict[str, float]:
        current_ratio = self.teacher_force_ratio

        # TODO: this value should be intelligently adjusted based on the loss/metrics/whatever
        # similar to what can be done for lr
        self.teacher_force_ratio = (self.teacher_force_ratio -
                                    self.teacher_force_drop) if self.teacher_force_ratio > self.teacher_force_drop else 0
        return {
            'teacher_force_ratio': current_ratio
        }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)

        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, cooldown=10),
            'interval': 'epoch',
            'monitor': 'train_loss/primary'
        }

        config = {
            'optimizer': optimizer,
            # 'lr_scheduler': lr_scheduler,
        }

        return config
