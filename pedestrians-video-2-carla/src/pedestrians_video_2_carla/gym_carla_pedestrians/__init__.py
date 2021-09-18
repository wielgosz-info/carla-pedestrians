import gym

# from https://github.com/mpSchrader/gym-sokoban/issues/29#issuecomment-612801318


def register(id, entry_point, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point,
    )


register(
    id='CarlaPedestrians-v0',
    entry_point='pedestrians_video_2_carla.gym_carla_pedestrians.envs:CarlaPedestriansEnv',
)
