"""
    This module contains the mock carla classes that are absolutely necessary for the rest of the code to work.
    It will work for modules that only need to import carla, but never execute actual CARLA-related code
    and for modules that are only setting basic Transforms, Locations and Rotations.
"""


class Transform(object):
    """
        This class is a mock of the carla.Transform class.
        It is only used for the rest of the code to work.
    """

    def __init__(self, location=None, rotation=None):
        self.location = location if location is not None else Location()
        self.rotation = rotation if rotation is not None else Rotation()


class Location(object):
    """
        This class is a mock of the carla.Location class.
        It is only used for the rest of the code to work.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class Rotation(object):
    """
        This class is a mock of the carla.Rotation class.
        It is only used for the rest of the code to work.
    """

    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
