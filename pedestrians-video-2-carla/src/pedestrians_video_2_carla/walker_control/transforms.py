from typing import Any, Dict, List
import carla
import pprint


def unreal_to_carla(unreal_transforms: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, carla.Transform]:
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


def relative_to_absolute(
        relative_transforms: Dict[str, carla.Transform],
        structure: List[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, carla.Transform]:
    # TODO: figure out this one day, the agent will probably work better in relative coords
    absolute_transforms = {}

    def transform_descendants(substructure, prev_transform: carla.Transform):
        # we shouldn't have more than one item here
        (bone_name, subsubstructures) = list(substructure.items())[0]
        absolute_transforms[bone_name] = carla.Transform(
            location=prev_transform.transform(relative_transforms[bone_name].location),
            rotation=carla.Rotation(
                pitch=prev_transform.rotation.pitch +
                relative_transforms[bone_name].rotation.pitch,
                yaw=prev_transform.rotation.yaw +
                relative_transforms[bone_name].rotation.yaw,
                roll=prev_transform.rotation.roll +
                relative_transforms[bone_name].rotation.roll
            )
        )
        if subsubstructures is not None:
            for subsubstructure in subsubstructures:
                transform_descendants(subsubstructure, absolute_transforms[bone_name])

    # we can only handle single root node
    transform_descendants(structure[0], carla.Transform())

    return absolute_transforms
