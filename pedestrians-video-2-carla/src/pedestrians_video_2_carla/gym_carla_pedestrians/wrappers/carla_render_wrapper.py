# TODO: rewrite to reuse CarlaRenderer from pedestrians_video_2_carla.renderers.carla_renderer

import copy
import logging
from collections import OrderedDict
from queue import Empty, Queue
from typing import Any

import carla
import gym
import PIL
import numpy as np
from pedestrians_video_2_carla.carla_utils.destroy import destroy_client_and_world
from pedestrians_video_2_carla.carla_utils.setup import *
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian


class CarlaRenderWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, fps=30.0, *args, **kwargs) -> None:
        super().__init__(env, *args, **kwargs)
        self.metadata['render.modes'].append('rgb_array')
        self.metadata['render.modes'] = list(set(self.metadata['render.modes']))
        self.metadata['video.frames_per_second'] = fps

        self._client: carla.Client = None
        self._world: carla.World = None
        self._sensor_dict: OrderedDict = None
        self._camera_queue: Queue = None

        self._fps = fps
        self._image_size = None

        self._bound_pedestrian = None

        self._logger = logging.getLogger(
            '{}.[{}]'.format(__name__, self.unwrapped.env_id))

    def close(self) -> None:
        super().close()

        if (self._client is not None) and (self._world is not None) and (self._sensor_dict is not None):
            destroy_client_and_world(self._client, self._world, self._sensor_dict)

    def reset(self, **kwargs) -> Any:
        self.close()

        observations = super().reset(**kwargs)

        self._client, self._world = setup_client_and_world(fps=self._fps)

        self._bound_pedestrian: ControlledPedestrian = copy.deepcopy(
            self.unwrapped.pedestrian)
        self._bound_pedestrian.bind(self._world)

        self._sensor_dict = OrderedDict()
        self._camera_queue = Queue()

        self._sensor_dict['camera_rgb'] = setup_camera(
            self._world, self._camera_queue, self._bound_pedestrian)

        self._image_size = (
            int(self._sensor_dict['camera_rgb'].attributes['image_size_x']),
            int(self._sensor_dict['camera_rgb'].attributes['image_size_y'])
        )

        return observations

    def render(self, mode='human', **kwargs):
        if mode == 'rgb_array':
            # sync current pedestrian pose & transform
            self._bound_pedestrian.current_pose.relative = self.unwrapped.pedestrian.current_pose.relative
            self._bound_pedestrian.apply_pose()

            self._bound_pedestrian.teleport_by(
                self.unwrapped.pedestrian.transform, False, True)

            world_frame = self._world.tick()

            frames = []
            sensor_data = None

            if world_frame:
                # drain the sensor queue
                try:
                    while (sensor_data is None) or sensor_data.frame < world_frame:
                        sensor_data = self._camera_queue.get(True, 1.0)
                        frames.append(sensor_data)
                except Empty:
                    self._logger.debug(
                        "Some sensor information is missed in frame {:06d}".format(world_frame))

                if len(frames):
                    data = frames[-1]
                    data.convert(carla.ColorConverter.Raw)
                    img = PIL.Image.frombuffer('RGBA', (data.width, data.height),
                                               data.raw_data, "raw", 'RGBA', 0, 1)  # load
                    img = img.convert('RGB')  # drop alpha
                    # the data is actually in BGR format, so switch channels
                    return np.array(img)[..., ::-1]

            return np.zeros((*self._image_size, 3), dtype=np.uint8)
        else:
            return self.env.render(mode, **kwargs)
