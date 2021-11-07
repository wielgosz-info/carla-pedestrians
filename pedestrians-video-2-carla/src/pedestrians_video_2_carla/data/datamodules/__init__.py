from .jaad_openpose import JAADOpenPoseDataModule
from .carla_2d_3d import Carla2D3DDataModule

DATA_MODULES = {}
DATA_MODULES['JAADOpenPose'] = JAADOpenPoseDataModule
DATA_MODULES['Carla2D3D'] = Carla2D3DDataModule
