import pytest


@pytest.fixture(scope="module")
def test_logs_dir():
    """
    Create a directory for the test logs.
    """
    import tempfile
    logs_dir = tempfile.mkdtemp()
    yield logs_dir
    import shutil
    shutil.rmtree(logs_dir)
