import os
from typing import Any, Dict, Pattern
import yaml
import json
import glob
from collections import OrderedDict

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
            "child_female": 'sk_girl_relative.yaml',
            "child_male": 'sk_kid_relative.yaml',
            "structure": 'structure.yaml',
        }[type]
    except KeyError:
        filename = type

    with open(os.path.join(os.path.dirname(__file__), '..', 'reference_skeletons', filename), 'r') as f:
        return yaml.load(f, Loader=Loader)


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
