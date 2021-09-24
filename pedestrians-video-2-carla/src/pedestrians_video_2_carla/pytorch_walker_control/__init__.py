"""
This module contains API-compatible classes to the ones in pedestrians_video_2_carla.walker_control,
but with calculations done with PyTorch and implementing the relevant `.forward()` methods.

Please note that input/output conventions (e.g., euler angles vs rotation matrices, direction of the Z axis etc.)
are not directly compatible between API methods (which are prepared in a way that allows relatively seamless
integration with CARLA) and `.forward()` (and related) methods, which are coded with PyTorch training flow
in mind.
"""
