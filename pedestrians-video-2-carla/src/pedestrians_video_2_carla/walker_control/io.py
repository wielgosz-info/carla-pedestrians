import os
from pprint import pprint
from typing import Any, Dict
import carla
import yaml

from pedestrians_video_2_carla.walker_control.transforms import relative_to_absolute_pose, unreal_to_carla

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_reference(type) -> Dict[str, Any]:
    try:
        filename = {
            "adult_female": 'sk_female_relative.yaml',
            "adult_male": 'sk_male_relative.yaml',
            "child_female": 'sk_girl_relative.yaml',
            "child_male": 'sk_kid_relative.yaml',
            "structure": 'structure.yaml',
        }[type]
    except KeyError:
        filename = type

    with open(os.path.join(os.path.dirname(__file__), '..', 'reference_skeletons', filename), 'r') as f:
        return yaml.load(f, Loader=Loader)
