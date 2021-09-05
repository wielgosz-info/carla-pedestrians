from collections import OrderedDict
import numpy as np
import gym


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
                ['crl_root', 'crl_hips__C', 'crl_spine__C', 'crl_spine01__C', 'crl_shoulder__L', 'crl_arm__L', 'crl_foreArm__L', 'crl_hand__L', 'crl_neck__C', 'crl_Head__C', 'crl_eye__L', 'crl_eye__R', 'crl_shoulder__R',
                    'crl_arm__R', 'crl_foreArm__R', 'crl_hand__R', 'crl_thigh__R', 'crl_leg__R', 'crl_foot__R', 'crl_toe__R', 'crl_toeEnd__R', 'crl_thigh__L', 'crl_leg__L', 'crl_foot__L', 'crl_toe__L', 'crl_toeEnd__L'],
                action[2:, :]  # [pitch, yaw, roll] * 26
            ))
        })
