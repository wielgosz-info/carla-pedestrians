from .linear import Linear
from .lstm import LSTM
from .linear_autoencoder import LinearAE
from .baseline_3d_pose import Baseline3DPose
from .linear_ae_residual import LinearAEResidual
from .baseline_3d_pose_rot import Baseline3DPoseRot
from .seq2seq import Seq2Seq
from .linear_ae_residual_tanh import LinearAEResidualTanh
from .seq2seq_embeddings import Seq2SeqEmbeddings
from .pose_former import PoseFormer

MOVEMENTS_MODELS = {
    m.__name__: m
    for m in [
        Linear,
        Baseline3DPose,
        Baseline3DPoseRot,
        LSTM,
        LinearAE,
        LinearAEResidual,
        Seq2Seq,
        LinearAEResidualTanh,
        Seq2SeqEmbeddings,
        PoseFormer
    ]
}
