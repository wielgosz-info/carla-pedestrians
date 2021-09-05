from collections import OrderedDict
import logging
from queue import Empty, Queue
import random

import carla
import gym
from gym import spaces
import numpy as np
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.pose import Pose
from pedestrians_video_2_carla.walker_control.pose_projection import PoseProjection


class CarlaPedestriansEnv(gym.Env):
    def __init__(self, env_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        bone_names = list(Pose().empty.keys())
        self.action_space = spaces.dict.Dict({
            'teleport_by': spaces.dict.Dict({
                # TODO: set some reasonable limits on teleportation?
                'location': spaces.Box(
                    low=np.array([-0.1, -0.1, 0]),
                    high=np.array([0.1, 0.1, 0.1]),
                    shape=(3,)),
                # we only allow yaw rotation:
                'rotation': spaces.Box(low=-180.0, high=180.0, shape=(1,))
            }),
            'update_pose': spaces.dict.Dict({
                # TODO: set some reasonable limits on bone rotations?
                bone_name: spaces.Box(low=-18.0, high=18.0, shape=(3,))
                for bone_name in bone_names
            }),
        })
        self.observation_space = spaces.dict.Dict({
            'relative_pose': spaces.dict.Dict({
                bone_name: spaces.dict.Dict({
                    'location': spaces.Box(low=-1.0, high=1.0, shape=(3,)),
                    'rotation': spaces.Box(low=-180.0, high=180.0, shape=(3,))
                })
                for bone_name in bone_names
            }),
            'absolute_pose': spaces.dict.Dict({
                bone_name: spaces.dict.Dict({
                    'location': spaces.Box(low=-4.0, high=4.0, shape=(3,)),
                    'rotation': spaces.Box(low=-180.0, high=180.0, shape=(3,))
                })
                for bone_name in bone_names
            }),
            'pose_projection': spaces.Box(low=0, high=800, shape=(26, 2)),
        })
        self.reward_range = (-np.inf, np.inf)

        self._env_id = env_id
        self._pedestrian: ControlledPedestrian = None
        self._pose_projection: PoseProjection = None

        self._logger = logging.getLogger('{}[{}]'.format(__name__, self._env_id))

    def _get_observation(self):
        return OrderedDict({
            'relative_pose': self._pedestrian.current_pose.relative,
            'absolute_pose': self._pedestrian.current_pose.absolute,
            'pose_projection': self._pose_projection.current_pose_to_points()
        })

    def seed(self, seed=None):
        # the ControlledPedestrian randomly selects the pedestrian's blueprint
        random.seed(seed)

        return [seed]

    def reset(self, age='adult', gender='female', initial_teleport: carla.Transform = None):
        # actually, we only need CARLA for rendering, so a lot of this could probably be moved to
        # some kind of rendering wrapper
        self.close()

        self._pedestrian = ControlledPedestrian(None, age, gender)
        self._pose_projection = PoseProjection(self._pedestrian)

        if initial_teleport is not None:
            self._pedestrian.teleport_by(initial_teleport, True)

        return self._get_observation()

    def step(self, action):
        self._pedestrian.teleport_by(carla.Transform(
            location=carla.Location(*action['teleport_by']['location'].astype(float)),
            rotation=carla.Rotation(
                yaw=action['teleport_by']['rotation'].astype(float)[0])
        ))
        self._pedestrian.update_pose({
            bone_name: carla.Rotation(*rotation.astype(float))
            for bone_name, rotation in action['update_pose'].items()
        })

        observation = self._get_observation()
        reward = 0.0
        done = False
        info = {
            'pedestrian': self._pedestrian,
            'pose_projection': self._pose_projection
        }

        return (
            observation,
            reward,
            done,
            info
        )
