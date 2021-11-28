"""
This module contains movements models based on Seq2Seq in various combinations.

They are based on the code from [Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)
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

from .seq2seq import Seq2Seq
from .seq2seq_embeddings import Seq2SeqEmbeddings
from .seq2seq_residual_a import Seq2SeqResidualA
from .seq2seq_residual_b import Seq2SeqResidualB
from .seq2seq_residual_c import Seq2SeqResidualC
