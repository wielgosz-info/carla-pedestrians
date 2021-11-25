import numpy as np
import warnings

try:
    import carla
except ImportError:
    import pedestrians_video_2_carla.carla_utils.mock_carla as carla
    warnings.warn("Using mock carla.", ImportWarning)


def test_unbound_transform(pedestrian):
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


def test_unbound_world_transform(pedestrian):
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


def test_bound_transform(carla_pedestrian):
    carla_pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ), True)

    assert np.isclose(carla_pedestrian.transform.location.x, 0.5)
    assert np.isclose(carla_pedestrian.transform.location.y, 0.5)
    # we cannot test Z due to uneven world surface
    assert np.isclose(carla_pedestrian.transform.rotation.pitch, 0)
    assert np.isclose(carla_pedestrian.transform.rotation.yaw, -30)
    assert np.isclose(carla_pedestrian.transform.rotation.roll, 0)

    carla_pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ), True)

    assert np.isclose(carla_pedestrian.transform.location.x, 1)
    assert np.isclose(carla_pedestrian.transform.location.y, 1)
    # we cannot test Z due to uneven world surface
    assert np.isclose(carla_pedestrian.transform.rotation.pitch, 0)
    assert np.isclose(carla_pedestrian.transform.rotation.yaw, -60)
    assert np.isclose(carla_pedestrian.transform.rotation.roll, 0)


def test_bound_world_transform(carla_pedestrian):
    carla_pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ), True)

    assert np.isclose(carla_pedestrian.world_transform.location.x,
                      carla_pedestrian.initial_transform.location.x + 0.5)
    assert np.isclose(carla_pedestrian.world_transform.location.y,
                      carla_pedestrian.initial_transform.location.y + 0.5)
    # we cannot test Z due to uneven world surface
    assert np.isclose(carla_pedestrian.world_transform.rotation.pitch,
                      carla_pedestrian.initial_transform.rotation.pitch)
    assert np.isclose(carla_pedestrian.world_transform.rotation.yaw,
                      carla_pedestrian.initial_transform.rotation.yaw + -30)
    assert np.isclose(carla_pedestrian.world_transform.rotation.roll,
                      carla_pedestrian.initial_transform.rotation.roll)

    carla_pedestrian.teleport_by(carla.Transform(
        location=carla.Location(
            x=0.5,
            y=0.5
        ),
        rotation=carla.Rotation(
            yaw=-30
        )
    ), True)

    assert np.isclose(carla_pedestrian.world_transform.location.x,
                      carla_pedestrian.initial_transform.location.x + 1)
    assert np.isclose(carla_pedestrian.world_transform.location.y,
                      carla_pedestrian.initial_transform.location.y + 1)
    # we cannot test Z due to uneven world surface
    assert np.isclose(carla_pedestrian.world_transform.rotation.pitch,
                      carla_pedestrian.initial_transform.rotation.pitch)
    assert np.isclose(carla_pedestrian.world_transform.rotation.yaw,
                      carla_pedestrian.initial_transform.rotation.yaw + -60)
    assert np.isclose(carla_pedestrian.world_transform.rotation.roll,
                      carla_pedestrian.initial_transform.rotation.roll)
