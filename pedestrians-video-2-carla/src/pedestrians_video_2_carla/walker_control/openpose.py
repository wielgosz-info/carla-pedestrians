import glob
import json
import os
from collections import OrderedDict
from typing import List, Pattern

import numpy as np


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


def openpose_to_image_points(keypoints_2d: List[float], empty_pose: OrderedDict) -> np.ndarray:
    """
    Converts list of points from OpenPose JSON output into image-space coordinates.

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
    empty_pose['crl_hips__C'] = points[8, :2]
    empty_pose['crl_spine__C'] = [np.NaN, np.NaN]
    empty_pose['crl_spine01__C'] = [np.NaN, np.NaN]
    empty_pose['crl_shoulder__L'] = [np.NaN, np.NaN]
    empty_pose['crl_arm__L'] = points[5, :2]
    empty_pose['crl_foreArm__L'] = points[6, :2]
    empty_pose['crl_hand__L'] = points[7, :2]
    # No. 1 is actually the point between shoulders in OpenPose, so lower than CARLA one
    empty_pose['crl_neck__C'] = points[1, :2]
    empty_pose['crl_Head__C'] = points[0, :2]
    empty_pose['crl_shoulder__R'] = [np.NaN, np.NaN]
    empty_pose['crl_arm__R'] = points[2, :2]
    empty_pose['crl_foreArm__R'] = points[3, :2]
    empty_pose['crl_hand__R'] = points[4, :2]
    empty_pose['crl_eye__L'] = points[16, :2]
    empty_pose['crl_eye__R'] = points[15, :2]
    empty_pose['crl_thigh__R'] = points[9, :2]
    empty_pose['crl_leg__R'] = points[10, :2]
    empty_pose['crl_foot__R'] = points[11, :2]
    empty_pose['crl_toe__R'] = points[22, :2]
    empty_pose['crl_toeEnd__R'] = points[23, :2]
    empty_pose['crl_thigh__L'] = points[12, :2]
    empty_pose['crl_leg__L'] = points[13, :2]
    empty_pose['crl_foot__L'] = points[14, :2]
    empty_pose['crl_toe__L'] = points[19, :2]
    empty_pose['crl_toeEnd__L'] = points[20, :2]

    return np.array(list(empty_pose.values()))


def alternative_hips_neck(original_points: np.ndarray, empty_pose: OrderedDict) -> np.ndarray:
    """
    Modifies the copy of original projection points with alternative calculations
    for hips and neck points. Hips are calculated as a mean between thighs, and
    nec is calculated as a mean between shoulders.

    :param original_points: Pose points projected to 2D
    :type original_points: np.ndarray
    :param empty_pose: `Pose().empty` to ensure we modify correct points
    :type empty_pose: OrderedDict
    :return: Pose points projected to 2D with hips and neck points modified
    :rtype: np.ndarray
    """
    projection_points = np.copy(original_points)

    bone_names = list(empty_pose.keys())
    hips_idx = bone_names.index('crl_hips__C')
    thighR_idx = bone_names.index('crl_thigh__R')
    thighL_idx = bone_names.index('crl_thigh__L')
    projection_points[hips_idx] = np.mean(
        [projection_points[thighR_idx], projection_points[thighL_idx]],
        axis=0
    )

    neck_idx = bone_names.index('crl_neck__C')
    shoulderR_idx = bone_names.index('crl_shoulder__R')
    shoulderL_idx = bone_names.index('crl_shoulder__L')
    projection_points[neck_idx] = np.mean(
        [projection_points[shoulderR_idx], projection_points[shoulderL_idx]],
        axis=0
    )

    return projection_points, (hips_idx, neck_idx)
