import pytest
from pedestrians_video_2_carla.loggers.pedestrian.pedestrian_renderers import PedestrianRenderers

from pedestrians_video_2_carla.modules.loss import LossModes
from pedestrians_video_2_carla.modules.projection.projection import ProjectionTypes


@pytest.fixture()
def test_logs_dir():
    """
    Create a directory for the test logs.
    """
    import tempfile
    logs_dir = tempfile.mkdtemp()
    yield logs_dir
    import shutil
    shutil.rmtree(logs_dir)


@pytest.fixture(scope="session")
def test_outputs_dir():
    """
    Create a directory for the test logs.
    """
    import tempfile
    outputs_dir = tempfile.mkdtemp()
    yield outputs_dir
    import shutil
    shutil.rmtree(outputs_dir)


@pytest.fixture(params=list(LossModes.__members__.keys()))
def loss_mode(request, projection_type):
    supported = LossModes.get_supported_loss_modes(ProjectionTypes[projection_type])
    if LossModes[request.param] not in supported:
        pytest.skip("Loss mode {} not supported for projection type {}".format(
            request.param, projection_type))
    return request.param


# those renderers should always be available
@pytest.fixture(params=['input_points', 'projection_points', 'none'])
def renderer(request):
    return request.param


@pytest.fixture(params=list(ProjectionTypes.__members__.keys()))
def projection_type(request):
    return request.param
