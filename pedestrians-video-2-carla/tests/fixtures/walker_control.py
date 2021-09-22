import pytest

from pedestrians_video_2_carla.utils.destroy import destroy
from pedestrians_video_2_carla.utils.setup import setup_client_and_world
from pedestrians_video_2_carla.utils.unreal import load_reference, unreal_to_carla
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian
from pedestrians_video_2_carla.walker_control.pose import Pose
from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.walker_control.pose_projection import PoseProjection
from pedestrians_video_2_carla.pytorch_walker_control.pose_projection import P3dPoseProjection


@pytest.fixture(params=[Pose, P3dPose])
def pose_cls(request):
    return request.param


@pytest.fixture(params=[PoseProjection, P3dPoseProjection])
def pose_projection_cls(request):
    return request.param


@pytest.fixture
def pedestrian(pose_cls, device):
    """
    Returns unbound ControlledPedestrian
    """
    return ControlledPedestrian(None, 'adult', 'female', PoseCls=pose_cls, device=device)


@pytest.fixture()
def carla_world():
    client, world = setup_client_and_world()
    yield world
    destroy(client, world, {})


@pytest.fixture
def carla_pedestrian(carla_world, pose_cls, device):
    """
    Returns ControlledPedestrian bound to specific CARLA world instance
    """
    return ControlledPedestrian(carla_world, 'adult', 'female', PoseCls=pose_cls, device=device)


@pytest.fixture
def absolute_pose():
    unreal_abs_pose = load_reference('sk_female_absolute.yaml')
    return unreal_to_carla(unreal_abs_pose['transforms'])


@pytest.fixture
def relative_pose():
    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    return unreal_to_carla(unreal_rel_pose['transforms'])


@pytest.fixture
def reference_pose(pose_cls, relative_pose, device):
    p = pose_cls(device=device)
    p.relative = relative_pose
    return p
