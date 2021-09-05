import gym
import numpy as np


class CarlaPedestriansEnv(gym.Env):
    def __init__(self, env_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._env_id = env_id

        self.action_space
        self.observation_space
        self.reward_range = (-np.inf, np.inf)

    def close(self):
        pass

    def reset(self):
        observation = {}

        return observation

    def step(self, action):
        observation = {}
        reward = 0.0
        done = True
        info = {}

        return (
            observation,
            reward,
            done,
            info
        )
