import pytest

from pedestrians_video_2_carla.modules.loss import LossModes
from pedestrians_video_2_carla.modules.base.output_types import MovementsModelOutputType
from pedestrians_video_2_carla.modules.movements import MOVEMENTS_MODELS
from pedestrians_video_2_carla.modules.trajectory import TRAJECTORY_MODELS


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
def loss_mode(request, movements_output_type):
    supported = LossModes.get_supported_loss_modes(
        MovementsModelOutputType[movements_output_type])
    if LossModes[request.param] not in supported:
        pytest.skip("Loss mode {} not supported for projection type {}".format(
            request.param, movements_output_type))
    return request.param


# those renderers should always be available
@pytest.fixture(params=['input_points', 'projection_points', 'none'])
def renderer(request):
    return request.param


@pytest.fixture(params=list(MovementsModelOutputType.__members__.keys()))
def movements_output_type(request):
    return request.param


# all models should be able to run with default settings
@pytest.fixture(params=MOVEMENTS_MODELS.keys())
def movements_model_name(request):
    return request.param


# all models should be able to run with default settings
@pytest.fixture(params=TRAJECTORY_MODELS.keys())
def trajectory_model_name(request):
    return request.param
