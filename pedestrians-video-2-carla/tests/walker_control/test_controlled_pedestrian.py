import carla
import numpy as np
from pedestrians_video_2_carla.utils.destroy import destroy
from pedestrians_video_2_carla.utils.setup import setup_client_and_world
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian


def test_unbound_transform():
    pedestrian = ControlledPedestrian(None, 'adult', 'female')

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ))

    assert np.isclose(pedestrian.transform.location.x, 0.5)
    assert np.isclose(pedestrian.transform.location.y, 0.5)
    assert np.isclose(pedestrian.transform.location.z, 0)
    assert np.isclose(pedestrian.transform.rotation.pitch, 0)
    assert np.isclose(pedestrian.transform.rotation.yaw, -30)
    assert np.isclose(pedestrian.transform.rotation.roll, 0)

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ))

    assert np.isclose(pedestrian.transform.location.x, 1)
    assert np.isclose(pedestrian.transform.location.y, 1)
    assert np.isclose(pedestrian.transform.location.z, 0)
    assert np.isclose(pedestrian.transform.rotation.pitch, 0)
    assert np.isclose(pedestrian.transform.rotation.yaw, -60)
    assert np.isclose(pedestrian.transform.rotation.roll, 0)


def test_unbound_world_transform():
    pedestrian = ControlledPedestrian(None, 'adult', 'female')

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ))

    assert np.isclose(pedestrian.world_transform.location.x, 0.5)
    assert np.isclose(pedestrian.world_transform.location.y, 0.5)
    assert np.isclose(pedestrian.world_transform.location.z, 0)
    assert np.isclose(pedestrian.world_transform.rotation.pitch, 0)
    assert np.isclose(pedestrian.world_transform.rotation.yaw, -30)
    assert np.isclose(pedestrian.world_transform.rotation.roll, 0)

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ))

    assert np.isclose(pedestrian.world_transform.location.x, 1)
    assert np.isclose(pedestrian.world_transform.location.y, 1)
    assert np.isclose(pedestrian.world_transform.location.z, 0)
    assert np.isclose(pedestrian.world_transform.rotation.pitch, 0)
    assert np.isclose(pedestrian.world_transform.rotation.yaw, -60)
    assert np.isclose(pedestrian.world_transform.rotation.roll, 0)


def test_bound_transform():
    client, world = setup_client_and_world()
    pedestrian = ControlledPedestrian(world, 'adult', 'female')

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ), True)

    assert np.isclose(pedestrian.transform.location.x, 0.5)
    assert np.isclose(pedestrian.transform.location.y, 0.5)
    # we cannot test Z due to uneven world surface
    assert np.isclose(pedestrian.transform.rotation.pitch, 0)
    assert np.isclose(pedestrian.transform.rotation.yaw, -30)
    assert np.isclose(pedestrian.transform.rotation.roll, 0)

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ), True)

    assert np.isclose(pedestrian.transform.location.x, 1)
    assert np.isclose(pedestrian.transform.location.y, 1)
    # we cannot test Z due to uneven world surface
    assert np.isclose(pedestrian.transform.rotation.pitch, 0)
    assert np.isclose(pedestrian.transform.rotation.yaw, -60)
    assert np.isclose(pedestrian.transform.rotation.roll, 0)

    destroy(client, world, {})


def test_bound_world_transform():
    client, world = setup_client_and_world()
    pedestrian = ControlledPedestrian(world, 'adult', 'female')

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ), True)

    assert np.isclose(pedestrian.world_transform.location.x,
                      pedestrian.initial_transform.location.x + 0.5)
    assert np.isclose(pedestrian.world_transform.location.y,
                      pedestrian.initial_transform.location.y + 0.5)
    # we cannot test Z due to uneven world surface
    assert np.isclose(pedestrian.world_transform.rotation.pitch,
                      pedestrian.initial_transform.rotation.pitch)
    assert np.isclose(pedestrian.world_transform.rotation.yaw,
                      pedestrian.initial_transform.rotation.yaw + -30)
    assert np.isclose(pedestrian.world_transform.rotation.roll,
                      pedestrian.initial_transform.rotation.roll)

    pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ), True)

    assert np.isclose(pedestrian.world_transform.location.x,
                      pedestrian.initial_transform.location.x + 1)
    assert np.isclose(pedestrian.world_transform.location.y,
                      pedestrian.initial_transform.location.y + 1)
    # we cannot test Z due to uneven world surface
    assert np.isclose(pedestrian.world_transform.rotation.pitch,
                      pedestrian.initial_transform.rotation.pitch)
    assert np.isclose(pedestrian.world_transform.rotation.yaw,
                      pedestrian.initial_transform.rotation.yaw + -60)
    assert np.isclose(pedestrian.world_transform.rotation.roll,
                      pedestrian.initial_transform.rotation.roll)

    destroy(client, world, {})
