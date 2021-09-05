from gym.envs.registration import register

register(
    id='carla-pedestrians-v0',
    entry_point='gym_carla_pedestrians.envs:CarlaPedestriansEnv',
)
