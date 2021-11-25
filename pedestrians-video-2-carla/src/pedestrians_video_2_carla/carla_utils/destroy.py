import warnings

try:
    import carla
except ImportError:
    import pedestrians_video_2_carla.carla_utils.mock_carla as carla
    warnings.warn("Using mock carla.", ImportWarning)


def destroy_client_and_world(client, world, sensor_dict=None):
    if getattr(carla, 'World', None) is None:
        raise RuntimeError(
            "You are using mock carla, calls to destroy_client_and_world are not allowed!")

    all_actors = world.get_actors()

    if sensor_dict is not None:
        for sensor in sensor_dict.values():
            sensor.stop()
            sensor.destroy()

    for controller in all_actors.filter('controller.*.*'):
        controller.stop()

    client.apply_batch_sync(
        [carla.command.DestroyActor(actor.id) for actor in list(
            all_actors.filter('controller.*.*')) + list(
            all_actors.filter('walker.*.*')) + list(
            all_actors.filter('vehicle.*.*'))]
    )

    world.tick()

    world.apply_settings(carla.WorldSettings(
        synchronous_mode=False
    ))

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(False)

    world.tick()
