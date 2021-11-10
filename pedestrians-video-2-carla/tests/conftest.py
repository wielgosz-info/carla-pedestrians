"""
    Dummy conftest.py for pedestrians_video_2_carla.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest

pytest_plugins = [
    "tests.fixtures.torch",
    "tests.fixtures.walker_control",
    "tests.fixtures.flow",
]
