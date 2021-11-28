from torch import nn
from .linear_ae_residual import LinearAEResidual


class LinearAEResidualLeaky(LinearAEResidual):
    """
    Residual bottleneck autoencoder with LeakyReLU.
    Inputs are flattened to a vector of size (input_nodes_len * input_features).
    """

    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**{
            **kwargs,
            'activation_cls': nn.LeakyReLU
        })
