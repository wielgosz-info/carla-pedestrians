import os
import carla
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_reference(type):
    filename = {
        "adult_female": 'sk_female.yaml',
        "adult_male": 'sk_male.yaml',
        "child_female": 'sk_girl.yaml',
        "child_male": 'sk_kid.yaml',
        "structure": 'structure.yaml',
    }[type]

    with open(os.path.join(os.path.dirname(__file__), '..', 'reference_skeletons', filename), 'r') as f:
        return yaml.load(f, Loader=Loader)


def apply_reference_pose(world, pedestrian):
    age = pedestrian.attributes['age']
    gender = pedestrian.attributes['gender']

    relative_pose = load_reference('{}_{}'.format(age, gender))

    # TODO: the whole coordinates and relative -> absolute conversion
    absolute_pose = relative_pose['transforms']

    control = carla.WalkerBoneControl()
    control.bone_transforms = [
        (bone_name, carla.Transform(
            location=carla.Location(
                x=transform_dict['location']['x']/100.0,
                z=-transform_dict['location']['y']/100.0,
                y=transform_dict['location']['z']/100.0,
            ),
            rotation=carla.Rotation(
                pitch=transform_dict['rotation']['pitch'],
                yaw=transform_dict['rotation']['yaw'],
                roll=transform_dict['rotation']['roll'],
            )
        )) for (bone_name, transform_dict) in absolute_pose.items()
    ]
    pedestrian.apply_control(control)
    world.tick()
