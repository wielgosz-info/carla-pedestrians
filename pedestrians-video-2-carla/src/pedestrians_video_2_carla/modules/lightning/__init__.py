from .linear import LitLinearMapper
from .lstm import LitLSTMMapper
from .linear_autoencoder import LitLinearAutoencoderMapper
from .baseline_3d_pose import LitBaseline3DPoseMapper
from .linear_ae_residual import LitLinearAEResidual
from .baseline_3d_pose_rot import LitBaseline3DPoseRot

MODELS = {}
MODELS['Linear'] = LitLinearMapper
MODELS['LSTM'] = LitLSTMMapper
MODELS['LinearAutoencoder'] = LitLinearAutoencoderMapper
MODELS['Baseline3DPose'] = LitBaseline3DPoseMapper
MODELS['LitLinearAEResidual'] = LitLinearAEResidual
MODELS['LitBaseline3DPoseRot'] = LitBaseline3DPoseRot
