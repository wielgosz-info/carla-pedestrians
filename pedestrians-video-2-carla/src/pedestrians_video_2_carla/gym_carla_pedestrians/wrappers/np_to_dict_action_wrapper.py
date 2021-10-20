from collections import OrderedDict
import numpy as np
import gym

from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON


class NumpyToDictActionWrapper(gym.ActionWrapper):
    """
    Simple action wrapper that unpacks action formatted as np.ndarray
    and converts it to OrderedDict as expected by the env
    """

    def __init__(self, env):
        super().__init__(env)

    def action(self, action: np.ndarray) -> OrderedDict:
        """
        Maps action from numpy array to Action Space

        :param action: NumPy array with shape (28,3). The teleport_by.rotation.pitch and teleport_by.rotation.roll are ignored
        :type action: np.ndarray
        :return: Action in format expected by CarlaPedestriansEnv
        :rtype: OrderedDict
        """
        return OrderedDict({
            'teleport_by': {
                'location': action[0, :],  # x,y,z
                'rotation': action[1, 1:2]  # skip pitch, yaw, skip roll
            },
            'update_pose': dict(zip(
                [k.name for k in CARLA_SKELETON],
                action[2:, :]  # [pitch, yaw, roll] * 26
            ))
        })
