from typing import Any, Dict, List, OrderedDict, Union
import carla
import numpy as np
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


def openpose_to_image_points(keypoints_2d: List[float], pedestrian: Any) -> np.ndarray:
    """
    [summary]

    :param keypoints_2d: List of keypoints in OpenPose format
        `[x0,y0,confidence0, x1,y1,confidence1, ..., x24,y24,confidence24]`
    :type keypoints_2d: List[float]
    :param pedestrian: pedestrian object on which `structure_as_pose()` can be called to get an empty
        OrderedDict with bones in correct order
    :type pedestrian: ControlledPedestrian
    :return: array of points in the image coordinates
    :rtype: np.ndarray
    """
    structure_as_pose, _ = pedestrian.structure_as_pose()

    points = np.array(keypoints_2d).reshape((-1, 3))  # x,y,confidence

    # match OpenPose BODY_25 to CARLA walker bones as much as possible
    structure_as_pose['crl_root'] = [np.NaN, np.NaN]
    # No. 8 is actually the point between thighs in OpenPose, so lower than CARLA one
    structure_as_pose['crl_hips__C'] = points[8, :2]
    structure_as_pose['crl_spine__C'] = [np.NaN, np.NaN]
    structure_as_pose['crl_spine01__C'] = [np.NaN, np.NaN]
    structure_as_pose['crl_shoulder__L'] = [np.NaN, np.NaN]
    structure_as_pose['crl_arm__L'] = points[5, :2]
    structure_as_pose['crl_foreArm__L'] = points[6, :2]
    structure_as_pose['crl_hand__L'] = points[7, :2]
    # No. 1 is actually the point between shoulders in OpenPose, so lower than CARLA one
    structure_as_pose['crl_neck__C'] = points[1, :2]
    structure_as_pose['crl_Head__C'] = points[0, :2]
    structure_as_pose['crl_shoulder__R'] = [np.NaN, np.NaN]
    structure_as_pose['crl_arm__R'] = points[2, :2]
    structure_as_pose['crl_foreArm__R'] = points[3, :2]
    structure_as_pose['crl_hand__R'] = points[4, :2]
    structure_as_pose['crl_eye__L'] = points[16, :2]
    structure_as_pose['crl_eye__R'] = points[15, :2]
    structure_as_pose['crl_thigh__R'] = points[9, :2]
    structure_as_pose['crl_leg__R'] = points[10, :2]
    structure_as_pose['crl_foot__R'] = points[11, :2]
    structure_as_pose['crl_toe__R'] = points[22, :2]
    structure_as_pose['crl_toeEnd__R'] = points[23, :2]
    structure_as_pose['crl_thigh__L'] = points[12, :2]
    structure_as_pose['crl_leg__L'] = points[13, :2]
    structure_as_pose['crl_foot__L'] = points[14, :2]
    structure_as_pose['crl_toe__L'] = points[19, :2]
    structure_as_pose['crl_toeEnd__L'] = points[20, :2]

    return np.array(list(structure_as_pose.values()))


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
        relative_transforms: OrderedDict[str, carla.Transform],
        structure: List[Dict[str, List[Dict[str, Any]]]]) -> OrderedDict[str, carla.Transform]:
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


def normalize_points(image_points, not_nans, hips_idx):
    hips = image_points[hips_idx]
    points = image_points[not_nans]

    height = points[:, 1].max() - points[:, 1].min()
    zeros = np.array([
        hips[0],  # X of hips point
        points[:, 1].min()
    ])
    points = (points - zeros) / height

    return points


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
