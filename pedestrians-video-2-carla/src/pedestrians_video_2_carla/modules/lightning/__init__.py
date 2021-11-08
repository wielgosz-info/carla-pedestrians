from .linear import LitLinearMapper
from .lstm import LitLSTMMapper
from .linear_autoencoder import LitLinearAutoencoderMapper
from .baseline_3d_pose import LitBaseline3DPoseMapper
from .linear_ae_residual import LitLinearAEResidual

MODELS = {}
MODELS['Linear'] = LitLinearMapper
MODELS['LSTM'] = LitLSTMMapper
MODELS['LinearAutoencoder'] = LitLinearAutoencoderMapper
MODELS['Baseline3DPose'] = LitBaseline3DPoseMapper
MODELS['LitLinearAEResidual'] = LitLinearAEResidual
