import carla


def destroy(client, world, sensor_list):
    all_actors = world.get_actors()

    for sensor in sensor_list.values():
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
