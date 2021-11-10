from .linear import Linear
from .lstm import LSTM
from .linear_autoencoder import LinearAE
from .baseline_3d_pose import Baseline3DPose
from .linear_ae_residual import LinearAEResidual
from .baseline_3d_pose_rot import Baseline3DPoseRot

MODELS = {
    m.__name__: m
    for m in [
        Linear,
        Baseline3DPose,
        Baseline3DPoseRot,
        LSTM,
        LinearAE,
        LinearAEResidual
    ]
}
