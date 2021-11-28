from .linear import Linear
from .lstm import LSTM
from .linear_ae import LinearAE, LinearAEResidual, LinearAEResidualLeaky
from .baseline_3d_pose import Baseline3DPose, Baseline3DPoseRot
from .seq2seq import Seq2Seq, Seq2SeqEmbeddings, Seq2SeqResidualA, Seq2SeqResidualB, Seq2SeqResidualC
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
        LinearAEResidualLeaky,
        Seq2Seq,
        Seq2SeqEmbeddings,
        Seq2SeqResidualA,
        Seq2SeqResidualB,
        Seq2SeqResidualC,
        PoseFormer
    ]
}
