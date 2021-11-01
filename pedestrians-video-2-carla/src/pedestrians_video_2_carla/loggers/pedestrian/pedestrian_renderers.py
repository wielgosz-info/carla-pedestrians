from enum import Enum


class PedestrianRenderers(Enum):
    none = 0
    source_videos = 1
    source_carla = 2
    input_points = 3
    projection_points = 4
    carla = 5
