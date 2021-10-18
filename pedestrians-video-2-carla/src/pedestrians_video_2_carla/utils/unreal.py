import os
from typing import Any, Dict

import carla
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_reference(type: str) -> Dict[str, Any]:
    """
    Loads the file with reference pose extracted from UE4 engine.

    :param type: One of 'adult_female', 'adult_male', 'child_female', 'child_male', 'structure'
        or arbitrary file name (relative to `reference_skeletons` dir).
    :type type: str
    :return: Dictionary containing pose structure or transforms.
    :rtype: Dict[str, Any]
    """
    try:
        filename = {
            "adult_female": 'sk_female_relative.yaml',
            "adult_male": 'sk_male_relative.yaml',
            "senior_female": 'sk_female_relative.yaml',  # there are no separate models for this
            "senior_male": 'sk_male_relative.yaml',  # there are no separate models for this
            "child_female": 'sk_girl_relative.yaml',
            "child_male": 'sk_kid_relative.yaml',
            "structure": 'structure.yaml',
        }[type]
    except KeyError:
        filename = type

    with open(os.path.join(os.path.dirname(__file__), '..', 'reference_skeletons', filename), 'r') as f:
        return yaml.load(f, Loader=Loader)


def unreal_to_carla(unreal_transforms: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, carla.Transform]:
    """
    Convert dict with transforms read from unreal into carla-usable format.

    :param unreal_transforms: Data loaded using `load_reference()['transforms']`
    :type unreal_transforms: Dict[str, Dict[str, Dict[str, float]]]
    :return: Transforms data mapped to carla.Transform
    :rtype: Dict[str, carla.Transform]
    """
    return {
        bone_name: carla.Transform(
            location=carla.Location(
                x=transform_dict['location']['x']/100.0,
                y=transform_dict['location']['y']/100.0,
                z=transform_dict['location']['z']/100.0,
            ),
            rotation=carla.Rotation(
                pitch=transform_dict['rotation']['pitch'],
                yaw=transform_dict['rotation']['yaw'],
                roll=transform_dict['rotation']['roll'],
            )
        ) for (bone_name, transform_dict) in unreal_transforms.items()
    }
