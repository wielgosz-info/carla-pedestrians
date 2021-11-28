from typing import Tuple
import torch
from .seq2seq_embeddings import Seq2SeqEmbeddings


class Seq2SeqResidualA(Seq2SeqEmbeddings):
    """
    Sequence to sequence model with embeddings and residual connections version A.
    This version keeps the residual part in returned output.
    """

    def _decode_frame(self,
                      hidden: torch.Tensor,
                      cell: torch.Tensor,
                      input: torch.Tensor,
                      needs_forcing: bool,
                      force_indices: torch.Tensor,
                      target_pose_changes: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        # insert input token embedding, previous hidden and previous cell states
        # receive output tensor (predictions) and new hidden and cell states
        output, hidden, cell = self.decoder(input, hidden, cell)
        residual_output = output + input

        force_input = input[force_indices]
        input = residual_output

        if needs_forcing:
            input[force_indices] = target_pose_changes + force_input

        return input, residual_output
