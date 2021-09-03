from typing import Any, Dict, List, OrderedDict, Union
import carla
from scipy.spatial.transform import Rotation


def unreal_to_carla(unreal_transforms: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, carla.Transform]:
    """
    Convert dict with transforms read from unreal into carla-usable format
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


def carla_to_scipy_rotation(rotation: Union[carla.Rotation, carla.Transform]) -> Rotation:
    """
    Convert carla.Rotation or carla.Transform to scipy.spatial.transform.Rotation
    """

    if isinstance(rotation, carla.Transform):
        rotation = rotation.rotation

    # we need to follow UE4 order of axes
    return Rotation.from_euler('ZYX', [
        rotation.yaw,
        rotation.pitch,
        rotation.roll,
    ], degrees=True)


def scipy_to_carla_rotation(rotation: Rotation) -> carla.Rotation:
    (yaw, pitch, roll) = rotation.as_euler('ZYX', degrees=True)
    return carla.Rotation(
        pitch=pitch,
        yaw=yaw,
        roll=roll
    )


def mul_rotations(reference_rotation: carla.Rotation, local_rotation: carla.Rotation) -> carla.Rotation:
    reference_rot = carla_to_scipy_rotation(reference_rotation)
    local_rot = carla_to_scipy_rotation(local_rotation)

    # and now multiply & convert it back
    return scipy_to_carla_rotation(reference_rot*local_rot)


def relative_to_absolute_pose(
        relative_transforms: Dict[str, carla.Transform],
        structure: List[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, carla.Transform]:
    # ensure bones in absolute pose will be in the same order as they were in relative
    absolute_transforms = OrderedDict(relative_transforms)

    def transform_descendants(substructure, prev_transform: carla.Transform):
        # we shouldn't have more than one item here
        (bone_name, subsubstructures) = list(substructure.items())[0]

        # we need to manually copy the carla.Location
        # since it seems to be modified in place
        absolute_transforms[bone_name] = carla.Transform(
            location=prev_transform.transform(carla.Location(
                x=relative_transforms[bone_name].location.x,
                y=relative_transforms[bone_name].location.y,
                z=relative_transforms[bone_name].location.z
            )),
            rotation=mul_rotations(
                prev_transform.rotation, relative_transforms[bone_name].rotation)
        )
        if subsubstructures is not None:
            for subsubstructure in subsubstructures:
                transform_descendants(subsubstructure, absolute_transforms[bone_name])

    # we can only handle single root node
    transform_descendants(structure[0], carla.Transform())

    return absolute_transforms


if __name__ == '__main__':
    # test if converting relative to absolute pos works correctly
    def pprint_transform(transform: carla.Transform):
        pprint.pprint({
            "location": {
                "x": "{:.4f}".format(transform.location.x),
                "y": "{:.4f}".format(transform.location.y),
                "z": "{:.4f}".format(transform.location.z),
            },
            "rotation": {
                "pitch": "{:.4f}".format(transform.rotation.pitch),
                "yaw": "{:.4f}".format(transform.rotation.yaw),
                "roll": "{:.4f}".format(transform.rotation.roll),
            }
        })

    def pprint_structure(substructure, ref_pose, calc_pose):
        (bone_name, subsubstructures) = list(substructure.items())[0]

        pprint.pprint(bone_name)
        pprint_transform(ref_pose[bone_name])
        pprint_transform(calc_pose[bone_name])

        if subsubstructures is not None:
            for subsubstructure in subsubstructures:
                pprint_structure(subsubstructure, ref_pose, calc_pose)

    from pedestrians_video_2_carla.walker_control.io import load_reference
    import pprint

    unreal_pose = load_reference('sk_female_absolute.yaml')
    absolute_pose = unreal_to_carla(unreal_pose['transforms'])

    unreal_rel_pose = load_reference('sk_female_relative.yaml')
    relative_pose = unreal_to_carla(unreal_rel_pose['transforms'])

    structure = load_reference('structure')['structure']

    calculated_abs = relative_to_absolute_pose(relative_pose, structure)

    pprint_structure(structure[0], absolute_pose, calculated_abs)
