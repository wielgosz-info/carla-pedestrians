import carla


def destroy(client, world):
    all_actors = world.get_actors()

    for sensor in all_actors.filter('sensor.*.*'):
        sensor.stop()

    for controller in all_actors.filter('controller.*.*'):
        controller.stop()

    results = client.apply_batch_sync(
        [carla.command.DestroyActor(actor.id) for actor in list(all_actors.filter('sensor.*.*')) + list(all_actors.filter(
            'controller.*.*')) + list(all_actors.filter('walker.*.*')) + list(all_actors.filter('vehicle.*.*'))]
    )

    print(results)

    world.tick()
