import carla
import cameratransform as ct

from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian


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

    world.tick()

    return client, world


def setup_camera(world, sensor_queue, pedestrian):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    pedestrian_transform = pedestrian.transform
    camera_rgb = world.spawn_actor(camera_bp, carla.Transform(
        carla.Location(
            x=pedestrian_transform.location.x+3.1,
            y=pedestrian_transform.location.y,
            z=pedestrian_transform.location.z
        ),
        carla.Rotation(
            pitch=pedestrian_transform.rotation.pitch,
            yaw=pedestrian_transform.rotation.yaw-180,
            roll=pedestrian_transform.rotation.roll,
        )
    ))

    world.tick()

    def camera_callback(sensor_data, sensor_queue, sensor_name):
        sensor_data.save_to_disk(
            '/outputs/carla/{:06d}.png'.format(sensor_data.frame))
        sensor_queue.put((sensor_data.frame, sensor_name))

    camera_rgb.listen(lambda data: camera_callback(data, sensor_queue, "camera_rgb"))

    return camera_rgb
