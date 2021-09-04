from queue import Queue
import carla
from typing import Any, Tuple


def setup_client_and_world(fps=30.0) -> Tuple[carla.Client, carla.World]:
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


def get_camera_transform(pedestrian: Any, distance=3.1, elevation=1.2) -> carla.Transform:
    """
    Calculates the transform in front of the pedestrian.

    :param pedestrian: pedestrian that should be observed by the camera
    :type pedestrian: ControlledPedestrian
    :param distance: distance in meters from the root of pedestrian to camera, defaults to 3.1
    :type distance: float, optional
    :param elevation: camera elevation, defaults to 1.2
    :type elevation: float, optional
    :return: desired transform of the camera
    :rtype: carla.Transform
    """
    pedestrian_transform = pedestrian.world_transform
    return carla.Transform(
        carla.Location(
            x=pedestrian_transform.location.x-pedestrian.spawn_shift.x+distance,
            y=pedestrian_transform.location.y-pedestrian.spawn_shift.y,
            z=pedestrian_transform.location.z-pedestrian.spawn_shift.z+elevation
        ),
        carla.Rotation(
            pitch=pedestrian_transform.rotation.pitch,
            yaw=pedestrian_transform.rotation.yaw-180,
            roll=pedestrian_transform.rotation.roll,
        )
    )


def setup_camera(world: carla.World, sensor_queue: Queue, pedestrian: Any) -> carla.Sensor:
    """
    Sets up the camera with callback saving frames to disk in front of the pedestrian.

    :param world: 
    :type world: carla.World
    :param sensor_queue: 
    :type sensor_queue: Queue
    :param pedestrian: 
    :type pedestrian: ControlledPedestrian
    :return: RGB Camera
    :rtype: carla.Sensor
    """
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_tr = get_camera_transform(pedestrian)
    camera_rgb = world.spawn_actor(camera_bp, camera_tr)

    world.tick()

    def camera_callback(sensor_data, sensor_queue, sensor_name):
        sensor_data.save_to_disk(
            '/outputs/carla/{:06d}.png'.format(sensor_data.frame))
        sensor_queue.put((sensor_data.frame, sensor_name))

    camera_rgb.listen(lambda data: camera_callback(data, sensor_queue, "camera_rgb"))

    return camera_rgb
