import glob
import json
import os
from collections import OrderedDict
from enum import Enum
from typing import List, Pattern, Union

import numpy as np
import torch
from torch.functional import Tensor

from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON


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


# TODO: add weights?
COMMON_NODES = {
    'CARLA_2_BODY_25': [
        (CARLA_SKELETON.crl_hips__C, BODY_25.hips__C),
        (CARLA_SKELETON.crl_arm__L, BODY_25.arm__L),
        (CARLA_SKELETON.crl_foreArm__L, BODY_25.foreArm__L),
        (CARLA_SKELETON.crl_hand__L, BODY_25.hand__L),
        (CARLA_SKELETON.crl_neck__C, BODY_25.neck__C),
        (CARLA_SKELETON.crl_Head__C, BODY_25.head__C),
        (CARLA_SKELETON.crl_arm__R, BODY_25.arm__R),
        (CARLA_SKELETON.crl_foreArm__R, BODY_25.foreArm__R),
        (CARLA_SKELETON.crl_hand__R, BODY_25.hand__R),
        (CARLA_SKELETON.crl_eye__L, BODY_25.eye__L),
        (CARLA_SKELETON.crl_eye__R, BODY_25.eye__R),
        (CARLA_SKELETON.crl_thigh__R, BODY_25.thigh__R),
        (CARLA_SKELETON.crl_leg__R, BODY_25.leg__R),
        (CARLA_SKELETON.crl_foot__R, BODY_25.foot__R),
        (CARLA_SKELETON.crl_toe__R, BODY_25.toe__R),
        (CARLA_SKELETON.crl_toeEnd__R, BODY_25.toeEnd__R),
        (CARLA_SKELETON.crl_thigh__L, BODY_25.thigh__L),
        (CARLA_SKELETON.crl_leg__L, BODY_25.leg__L),
        (CARLA_SKELETON.crl_foot__L, BODY_25.foot__L),
        (CARLA_SKELETON.crl_toe__L, BODY_25.toe__L),
        (CARLA_SKELETON.crl_toeEnd__L, BODY_25.toeEnd__L),
    ],
    'CARLA_2_COCO': [
        (CARLA_SKELETON.crl_arm__L, COCO.arm__L),
        (CARLA_SKELETON.crl_foreArm__L, COCO.foreArm__L),
        (CARLA_SKELETON.crl_hand__L, COCO.hand__L),
        (CARLA_SKELETON.crl_neck__C, COCO.neck__C),
        (CARLA_SKELETON.crl_Head__C, COCO.head__C),
        (CARLA_SKELETON.crl_arm__R, COCO.arm__R),
        (CARLA_SKELETON.crl_foreArm__R, COCO.foreArm__R),
        (CARLA_SKELETON.crl_hand__R, COCO.hand__R),
        (CARLA_SKELETON.crl_eye__L, COCO.eye__L),
        (CARLA_SKELETON.crl_eye__R, COCO.eye__R),
        (CARLA_SKELETON.crl_thigh__R, COCO.thigh__R),
        (CARLA_SKELETON.crl_leg__R, COCO.leg__R),
        (CARLA_SKELETON.crl_foot__R, COCO.foot__R),
        (CARLA_SKELETON.crl_thigh__L, COCO.thigh__L),
        (CARLA_SKELETON.crl_leg__L, COCO.leg__L),
        (CARLA_SKELETON.crl_foot__L, COCO.foot__L),
    ]
}


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
