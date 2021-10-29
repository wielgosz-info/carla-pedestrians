# TODO: reuse PointsRenderer from pedestrians_video_2_carla.renderers.points_renderer

from typing import Any, Dict, List, Tuple
import gym
import numpy as np

from pedestrians_video_2_carla.walker_control.pose_projection import PoseProjection


class PoseOverlayRenderWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *args, **kwargs) -> None:
        super().__init__(env, *args, **kwargs)
        self.metadata['render.modes'].append('rgb_array')
        self.metadata['render.modes'] = list(set(self.metadata['render.modes']))

        self._last_pose_projection: np.ndarray = None
        self._pose_keys: List[str] = None

    def reset(self, **kwargs) -> Any:
        observation = super().reset(**kwargs)

        self._last_pose_projection = np.round(
            observation['pose_projection']).astype(int)
        self._pose_keys = self.unwrapped.pedestrian.current_pose.empty.keys()

        return observation

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        (
            observation,
            reward,
            done,
            info
        ) = super().step(action)

        self._last_pose_projection = np.round(
            observation['pose_projection']).astype(int)

        return (
            observation,
            reward,
            done,
            info
        )

    def render(self, mode='human', **kwargs):
        frame = super().render(mode, **kwargs)

        if (mode == 'rgb_array') and (frame is not None) and (self._last_pose_projection is not None):
            return PoseProjection.draw_projection_points(frame, self._last_pose_projection, self._pose_keys)

        return frame
