from pedestrians_video_2_carla.modules.trajectory.zero import ZeroTrajectory


TRAJECTORY_MODELS = {
    m.__name__: m
    for m in [
        ZeroTrajectory
    ]
}
