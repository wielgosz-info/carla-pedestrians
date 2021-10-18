import glob
import json
import os
from collections import OrderedDict
from enum import Enum
from typing import List, Pattern, Union

import numpy as np
import torch
from torch.functional import Tensor


class BODY_25(Enum):
    head__C = 0
    neck__C = 1
    arm__R = 2
    foreArm__R = 3
    hand__R = 4
    arm__L = 5
    foreArm__L = 6
    hand__L = 7
    hips__C = 8
    thigh__R = 9
    leg__R = 10
    foot__R = 11
    thigh__L = 12
    leg__L = 13
    foot__L = 14
    eye__R = 15
    eye__L = 16
    ear__R = 17
    ear_L = 18
    toe__L = 19
    toeEnd__L = 20
    heel_L = 21
    toe__R = 22
    toeEnd__R = 23
    heel_R = 24


class COCO(Enum):
    head__C = 0
    neck__C = 1
    arm__R = 2
    foreArm__R = 3
    hand__R = 4
    arm__L = 5
    foreArm__L = 6
    hand__L = 7
    thigh__R = 8
    leg__R = 9
    foot__R = 10
    thigh__L = 11
    leg__L = 12
    foot__L = 13
    eye__R = 14
    eye__L = 15
    ear__R = 16
    ear_L = 17


def load_openpose(path: str, frame_no_regexp: Pattern = None, frame_no_as_int=False):
    """
    Read a single OpenPose keypoints file or a set from it from the dir.
    When frame_no_reqexp is not specified, the files wil be sorted alphabetically
    and the frames numbered starting from 0.

    **Important note**: The order of returned poses is determined per-frame. There is no
    guarantee that the pose with index 0 in frame A and the pose with index 0 in frame B
    belong to the same person.

    :param path: Path ot dir containing *_keypoints.json files or to a single file.
    :type path: str
    :param frame_no_regexp: RegExp Pattern that should contain a single group that will be used
        as the frame number. Please noe that this is supposed to be the output of `re.compile`, so any flags 
        (like ignore case) need to be specified there; defaults to None
    :type frame_no_regexp: Pattern, optional
    :param frame_no_as_int: determines if the output of `frame_no_regexp` group fill be run through
        `int(output, 10)`; defaults to False
    :type frame_no_as_int: bool, optional

    :return: OrderedDict with keys being frame numbers and values being OpenPose keypoints list for each frame
    :rtype: OrderedDict
    """

    if path.startswith(os.path.sep):
        abspath = path
    else:
        abspath = os.path.join(os.path.dirname(__file__), path)

    if os.path.isdir(abspath):
        files = sorted(glob.glob(os.path.join(abspath, '*_keypoints.json')))
    else:
        files = [abspath]

    frames = {}

    for idx, file in enumerate(files):
        frame_no = idx
        if frame_no_regexp is not None:
            match = frame_no_regexp.match(file)
            if match is not None:
                try:
                    frame_no = match.group(1)
                    if frame_no_as_int:
                        frame_no = int(frame_no, 10)
                except IndexError:
                    # fall back to idx
                    pass
        with open(file, 'r') as f:
            raw = json.load(f)
            frames[frame_no] = raw['people']

    # insert frames in the correct order
    ordered_frames = OrderedDict()
    for frame_no in sorted(frames.keys()):
        ordered_frames[frame_no] = frames[frame_no]

    return ordered_frames


def openpose_to_projection_points(keypoints_2d: List[float], empty_pose: OrderedDict) -> np.ndarray:
    """
    Converts list of points from OpenPose BODY_25 JSON output into image-space coordinates.

    :param keypoints_2d: List of keypoints in OpenPose format
        `[x0,y0,confidence0, x1,y1,confidence1, ..., x24,y24,confidence24]`
    :type keypoints_2d: List[float]
    :param empty_pose: an empty OrderedDict with bones in correct order, e.g. obtained from pedestrian.current_pose.empty
    :type empty_pose: OrderedDict
    :return: array of points in the image coordinates
    :rtype: np.ndarray
    """
    points = np.array(keypoints_2d).reshape((-1, 3))  # x,y,confidence

    # match OpenPose BODY_25 to CARLA walker bones as much as possible
    # we then will get empty_pose.values() to ensure the points are in correct order
    empty_pose['crl_root'] = [np.NaN, np.NaN]
    # No. 8 is actually the point between thighs in OpenPose, so lower than CARLA one
    empty_pose['crl_hips__C'] = points[BODY_25.hips__C.value, :2]
    empty_pose['crl_spine__C'] = [np.NaN, np.NaN]
    empty_pose['crl_spine01__C'] = [np.NaN, np.NaN]
    empty_pose['crl_shoulder__L'] = [np.NaN, np.NaN]
    empty_pose['crl_arm__L'] = points[BODY_25.arm__L.value, :2]
    empty_pose['crl_foreArm__L'] = points[BODY_25.foreArm__L.value, :2]
    empty_pose['crl_hand__L'] = points[BODY_25.hand__L.value, :2]
    # No. 1 is actually the point between shoulders in OpenPose, so lower than CARLA one
    empty_pose['crl_neck__C'] = points[BODY_25.neck__C.value, :2]
    empty_pose['crl_Head__C'] = points[BODY_25.head__C.value, :2]
    empty_pose['crl_shoulder__R'] = [np.NaN, np.NaN]
    empty_pose['crl_arm__R'] = points[BODY_25.arm__R.value, :2]
    empty_pose['crl_foreArm__R'] = points[BODY_25.foreArm__R.value, :2]
    empty_pose['crl_hand__R'] = points[BODY_25.hand__R.value, :2]
    empty_pose['crl_eye__L'] = points[BODY_25.eye__L.value, :2]
    empty_pose['crl_eye__R'] = points[BODY_25.eye__R.value, :2]
    empty_pose['crl_thigh__R'] = points[BODY_25.thigh__R.value, :2]
    empty_pose['crl_leg__R'] = points[BODY_25.leg__R.value, :2]
    empty_pose['crl_foot__R'] = points[BODY_25.foot__R.value, :2]
    empty_pose['crl_toe__R'] = points[BODY_25.toe__R.value, :2]
    empty_pose['crl_toeEnd__R'] = points[BODY_25.toeEnd__R.value, :2]
    empty_pose['crl_thigh__L'] = points[BODY_25.thigh__L.value, :2]
    empty_pose['crl_leg__L'] = points[BODY_25.leg__L.value, :2]
    empty_pose['crl_foot__L'] = points[BODY_25.foot__L.value, :2]
    empty_pose['crl_toe__L'] = points[BODY_25.toe__L.value, :2]
    empty_pose['crl_toeEnd__L'] = points[BODY_25.toeEnd__L.value, :2]

    return np.array(list(empty_pose.values()))


def alternative_hips_neck(original_points: Union[np.ndarray, Tensor], empty_pose: OrderedDict) -> Union[np.ndarray, Tensor]:
    """
    Modifies the copy of original projection points with alternative calculations
    for hips and neck points. Hips are calculated as a mean between thighs, and
    nec is calculated as a mean between shoulders.

    :param original_points: Pose points projected to 2D
    :type original_points: Union[np.ndarray, Tensor]
    :param empty_pose: `Pose().empty` to ensure we modify correct points
    :type empty_pose: OrderedDict
    :return: Pose points projected to 2D with hips and neck points modified
    :rtype: Union[np.ndarray, Tensor]
    """
    if isinstance(original_points, Tensor):
        projection_points = torch.clone(original_points)
    else:
        projection_points = np.copy(original_points)

    bone_names = list(empty_pose.keys())
    hips_idx = bone_names.index('crl_hips__C')
    thighR_idx = bone_names.index('crl_thigh__R')
    thighL_idx = bone_names.index('crl_thigh__L')
    projection_points[hips_idx] = projection_points[[
        thighR_idx, thighL_idx]].mean(axis=0)

    neck_idx = bone_names.index('crl_neck__C')
    shoulderR_idx = bone_names.index('crl_shoulder__R')
    shoulderL_idx = bone_names.index('crl_shoulder__L')
    projection_points[neck_idx] = projection_points[[
        shoulderR_idx, shoulderL_idx]].mean(axis=0)

    return projection_points, (hips_idx, neck_idx)
