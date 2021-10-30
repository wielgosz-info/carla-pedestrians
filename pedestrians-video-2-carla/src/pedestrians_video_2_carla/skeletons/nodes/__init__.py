from .carla import CARLA_SKELETON

SKELETONS = {
    'CARLA_SKELETON': CARLA_SKELETON,
}
MAPPINGS = {}


def get_skeleton_type_by_name(name):
    return SKELETONS[name]


def get_skeleton_name_by_type(skeleton):
    return list(SKELETONS.keys())[list(SKELETONS.values()).index(skeleton)]


def register_skeleton(name, skeleton, mapping):
    SKELETONS[name] = skeleton
    MAPPINGS[skeleton] = mapping
