from collections import OrderedDict
import random
from typing import Dict, Tuple, Any
import carla

from pedestrians_video_2_carla.utils.unreal import load_reference, unreal_to_carla
from pedestrians_video_2_carla.walker_control.pose import Pose


class ControlledPedestrian(object):
    def __init__(self, world: carla.World = None, age: str = 'adult', gender: str = 'female', *args, **kwargs):
        """
        Initializes the pedestrian that keeps track of its current pose.

        :param world: world object of the connected client, if not specified all calculations will be done
            on the client side and with no rendering; defaults to None
        :type world: carla.World, optional
        :param age: one of 'adult' or 'child'; defaults to 'adult'
        :type age: str, optional
        :param gender: one of 'male' or 'female'; defaults to 'female'
        :type gender: str, optional
        """
        super().__init__()

        self._world = world
        self._age = age
        self._gender = gender

        # spawn point (may be different than actual location the pedesrian has spawned, especially Z-wise);
        # if world is not specified this will always be point 0,0,0
        self._spawn_loc = carla.Location()
        if self._world is not None:
            self._walker = self._spawn_walker()
        else:
            self._walker = None

        self._current_pose = Pose()
        self._current_pose.relative = self._apply_reference_pose()

        if self._walker is not None:
            self._initial_transform = self._walker.get_transform()
            self._world_transform = self._walker.get_transform()
        else:
            self._initial_transform = carla.Transform()
            self._world_transform = carla.Transform()

    def structure_as_pose(self) -> Tuple[OrderedDict, Dict[str, Any]]:
        pose = OrderedDict()

        if self._structure is not None:
            root = self._structure
        else:
            root = load_reference('structure')['structure']

        def add_to_pose(structure):
            (bone_name, substructures) = list(structure.items())[0]
            pose[bone_name] = None
            if substructures is not None:
                for substructure in substructures:
                    add_to_pose(substructure)

        add_to_pose(root[0])

        return pose, root

    def _spawn_walker(self):
        blueprint_library = self._world.get_blueprint_library()
        matching_blueprints = [bp for bp in blueprint_library.filter("walker.pedestrian.*")
                               if bp.get_attribute('age') == self._age and bp.get_attribute('gender') == self._gender]
        walker_bp = random.choice(matching_blueprints)

        walker = None
        tries = 0
        while walker is None and tries < 10:
            tries += 1
            walker_loc = self._world.get_random_location_from_navigation()
            walker = self._world.try_spawn_actor(walker_bp, carla.Transform(walker_loc))

        if walker is None:
            raise RuntimeError("Couldn't spawn walker")
        else:
            self._spawn_loc = walker_loc

        self._world.tick()

        return walker

    def _apply_reference_pose(self):
        unreal_pose = load_reference('{}_{}'.format(self._age, self._gender))
        relative_pose = unreal_to_carla(unreal_pose['transforms'])

        self._current_pose.relative = relative_pose
        self.apply_pose(True)

        return relative_pose

    def teleport_by(self, transform: carla.Transform, cue_tick=False):
        old_world_transform = self.world_transform
        self._world_transform = carla.Transform(
            location=carla.Location(
                x=old_world_transform.location.x + transform.location.x,
                y=old_world_transform.location.y + transform.location.y,
                z=old_world_transform.location.z + transform.location.z
            ),
            rotation=carla.Rotation(
                pitch=old_world_transform.rotation.pitch + transform.rotation.pitch,
                yaw=old_world_transform.rotation.yaw + transform.rotation.yaw,
                roll=old_world_transform.rotation.roll + transform.rotation.roll
            )
        )

        if self._walker is not None:
            self._walker.set_transform(self._world_transform)

            if cue_tick:
                self._world.tick()

    def move(self, rotations: Dict[str, carla.Rotation], cue_tick=False):
        """
        Apply the movement specified as change in local rotations for selected bones.
        For example `pedestrian.apply_movement({ 'crl_foreArm__L': carla.Rotation(pitch=60) })`
        will make the arm bend in the elbow by 60deg around Y axis using its current rotation
        plane (which gives roughly 60deg bend around the global Z axis).
        """

        self._current_pose.move(rotations)
        self.apply_pose(cue_tick)

    def apply_pose(self, cue_tick=False):
        """
        Applies the current absolute pose to the carla.Walker if it exists.

        :param cue_tick: should carla.World.tick() be called after sending control; defaults to False
        :type cue_tick: bool, optional
        """
        if self._walker is not None:
            control = carla.WalkerBoneControl()
            control.bone_transforms = list(self._current_pose.absolute.items())

            self._walker.apply_control(control)

            if cue_tick:
                self._world.tick()

    @property
    def age(self) -> str:
        return self._age

    @property
    def gender(self) -> str:
        return self._gender

    @property
    def world_transform(self) -> carla.Transform:
        if self._walker is not None:
            # if possible, get it from CARLA
            # don't ask me why Z is is some 0.91m above the actual root
            return self._walker.get_transform()
        return self._world_transform

    @property
    def transform(self) -> carla.Transform:
        """
        Current pedestrian transform relative to the position it was spawned at
        """
        world_transform = self.world_transform
        return carla.Transform(
            location=carla.Location(
                x=world_transform.location.x - self._initial_transform.location.x,
                y=world_transform.location.y - self._initial_transform.location.y,
                z=world_transform.location.z - self._initial_transform.location.z
            ),
            rotation=carla.Rotation(
                pitch=world_transform.rotation.pitch - self._initial_transform.rotation.pitch,
                yaw=world_transform.rotation.yaw - self._initial_transform.rotation.yaw,
                roll=world_transform.rotation.roll - self._initial_transform.rotation.roll
            )
        )

    @property
    def current_pose(self) -> Pose:
        return self._current_pose

    @property
    def spawn_shift(self):
        """
        Difference between spawn point and actual spawn location
        """
        return carla.Location(
            x=self._initial_transform.location.x - self._spawn_loc.x,
            y=self._initial_transform.location.y - self._spawn_loc.y,
            z=self._initial_transform.location.z - self._spawn_loc.z
        )


if __name__ == "__main__":
    from queue import Queue, Empty
    from collections import OrderedDict

    from pedestrians_video_2_carla.utils.destroy import destroy
    from pedestrians_video_2_carla.utils.setup import *

    client, world = setup_client_and_world()
    pedestrian = ControlledPedestrian(world, 'adult', 'female')

    sensor_list = OrderedDict()
    sensor_queue = Queue()

    sensor_list['camera_rgb'] = setup_camera(world, sensor_queue, pedestrian)

    ticks = 0
    while ticks < 10:
        world.tick()
        w_frame = world.get_snapshot().frame

        try:
            for _ in range(len(sensor_list.values())):
                s_frame = sensor_queue.get(True, 1.0)

            ticks += 1
        except Empty:
            print("Some sensor information is missed in frame {:06d}".format(w_frame))

        # teleport/rotate pedestrian a bit to see if teleport_by is working
        pedestrian.teleport_by(carla.Transform(
            location=carla.Location(
                x=random.random()-0.5,
                y=random.random()-0.5
            ),
            rotation=carla.Rotation(
                yaw=random.random()*60-30
            )
        ))

        # apply some movement to the left arm to see apply_movement in action
        pedestrian.move({
            'crl_arm__L': carla.Rotation(yaw=-random.random()*15),
            'crl_foreArm__L': carla.Rotation(pitch=-random.random()*15)
        })

    destroy(client, world, sensor_list)
