import carla
import random


def setup_client_and_world(fps=30.0):
    client = carla.Client('server', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    world.apply_settings(carla.WorldSettings(
        synchronous_mode=True,
        fixed_delta_seconds=1.0/fps,  # match the FPS from JAAD videos
        deterministic_ragdolls=False
    ))

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    return client, world


def setup_pedestrian(world, age, gender):
    blueprint_library = world.get_blueprint_library()
    matching_blueprints = [bp for bp in blueprint_library.filter("walker.pedestrian.*")
                           if bp.get_attribute('age') == age and bp.get_attribute('gender') == gender]
    walker_bp = random.choice(matching_blueprints)

    pedestrian = None
    tries = 0
    while pedestrian is None and tries < 10:
        tries += 1
        walker_loc = world.get_random_location_from_navigation()
        pedestrian = world.spawn_actor(walker_bp, carla.Transform(walker_loc))

    if pedestrian is None:
        raise RuntimeError("Couldn't spawn pedestrian")

    return pedestrian


def setup_camera(world, pedestrian):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_rgb = world.spawn_actor(camera_bp, carla.Transform(
        carla.Location(x=1, y=1, z=1.5),
        carla.Rotation(yaw=-179)
    ), attach_to=pedestrian)

    return camera_rgb
