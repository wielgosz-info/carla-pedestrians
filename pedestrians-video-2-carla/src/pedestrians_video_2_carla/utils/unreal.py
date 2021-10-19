import os
from typing import Any, Dict

import carla
import yaml
from enum import Enum

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class CARLA_SKELETON(Enum):
    crl_root = 0
    crl_hips__C = 1
    crl_spine__C = 2
    crl_spine01__C = 3
    crl_shoulder__L = 4
    crl_arm__L = 5
    crl_foreArm__L = 6
    crl_hand__L = 7
    crl_neck__C = 8
    crl_Head__C = 9
    crl_eye__L = 10
    crl_eye__R = 11
    crl_shoulder__R = 12
    crl_arm__R = 13
    crl_foreArm__R = 14
    crl_hand__R = 15
    crl_thigh__R = 16
    crl_leg__R = 17
    crl_foot__R = 18
    crl_toe__R = 19
    crl_toeEnd__R = 20
    crl_thigh__L = 21
    crl_leg__L = 22
    crl_foot__L = 23
    crl_toe__L = 24
    crl_toeEnd__L = 25


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
