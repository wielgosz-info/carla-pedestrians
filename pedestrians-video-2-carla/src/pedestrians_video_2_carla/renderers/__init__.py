from enum import Enum


class AlphaBehavior(Enum):
    drop = 0
    blend = 1
    keep = 2


class MergingMethod(Enum):
    vertical = 0
    horizontal = 1
    square = 2
