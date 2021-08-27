import os
from pprint import pprint
from typing import Any, Dict
import carla
import yaml

from pedestrians_video_2_carla.walker_control.transforms import relative_to_absolute, unreal_to_carla

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_reference(type) -> Dict[str, Any]:
    filename = {
        "adult_female": 'sk_female.yaml',
        "adult_male": 'sk_male.yaml',
        "child_female": 'sk_girl.yaml',
        "child_male": 'sk_kid.yaml',
        "structure": 'structure.yaml',
    }[type]

    with open(os.path.join(os.path.dirname(__file__), '..', 'reference_skeletons', filename), 'r') as f:
        return yaml.load(f, Loader=Loader)


def apply_reference_pose(world: carla.World, pedestrian: carla.Walker):
    age = pedestrian.attributes['age']
    gender = pedestrian.attributes['gender']

    unreal_pose = load_reference('{}_{}'.format(age, gender))
    unreal_pose.update(load_reference('structure'))
    absolute_pose = unreal_to_carla(unreal_pose['transforms'])

    control = carla.WalkerBoneControl()
    control.bone_transforms = list(absolute_pose.items())

    pedestrian.apply_control(control)
    world.tick()
