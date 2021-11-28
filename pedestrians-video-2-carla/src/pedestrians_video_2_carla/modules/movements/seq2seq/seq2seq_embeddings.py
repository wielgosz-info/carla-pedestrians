"""
Based on the code from [Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)
by [Ben Trevett](https://github.com/bentrevett) licensed under [MIT License](https://github.com/bentrevett/pytorch-seq2seq/blob/master/LICENSE),
which itself is an implementation of the paper https://arxiv.org/abs/1409.3215:

```bibtex
@misc{sutskever2014sequence,
      title={Sequence to Sequence Learning with Neural Networks}, 
      author={Ilya Sutskever and Oriol Vinyals and Quoc V. Le},
      year={2014},
      eprint={1409.3215},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
"""

from typing import Dict
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
import torch
from torch import nn
from torch.functional import Tensor
from .seq2seq import Seq2Seq


class Seq2SeqEmbeddings(Seq2Seq):
    """
    Sequence to sequence model with embeddings.

    Based on the code from [Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)
    by [Ben Trevett](https://github.com/bentrevett) licensed under [MIT License](https://github.com/bentrevett/pytorch-seq2seq/blob/master/LICENSE),
    which itself is an implementation of the paper https://arxiv.org/abs/1409.3215:

    ```bibtex
    @misc{sutskever2014sequence,
        title={Sequence to Sequence Learning with Neural Networks}, 
        author={Ilya Sutskever and Oriol Vinyals and Quoc V. Le},
        year={2014},
        eprint={1409.3215},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    ```
    """

    def __init__(self,
                 single_joint_embeddings_size=64,
                 **kwargs):
        super().__init__(**{
            **kwargs,
            'input_features': single_joint_embeddings_size
        })

        self.single_joint_embeddings_size = single_joint_embeddings_size
        self.embeddings = nn.ModuleList([nn.Linear(2, self.single_joint_embeddings_size)
                                         for _ in range(len(self.input_nodes))])

        self._hparams = {
            **self._hparams,
            'single_joint_embeddings_size': self.single_joint_embeddings_size
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Seq2Seq.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("Seq2SeqEmbeddings Movements Module")
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

        needs_forcing, target_pose_changes, force_indices = self._teacher_forcing(
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
