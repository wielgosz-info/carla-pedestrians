from collections import OrderedDict
import copy
import random
from typing import Dict, Type
import carla
from pedestrians_video_2_carla.carla_utils.spatial import deepcopy_location, deepcopy_transform

from pedestrians_video_2_carla.skeletons.reference.load import load_reference, unreal_to_carla
from pedestrians_video_2_carla.walker_control.pose import Pose


class ControlledPedestrian(object):
    def __init__(self, world: carla.World = None, age: str = 'adult', gender: str = 'female', pose_cls: Type = Pose, max_spawn_tries=10, *args, **kwargs):
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

        self._age = age
        self._gender = gender
        self._current_pose: pose_cls = pose_cls(**kwargs)
        self._current_pose.relative = self._load_reference_pose()

        # spawn point (may be different than actual location the pedesrian has spawned, especially Z-wise);
        # if world is not specified this will always be point 0,0,0
        self._spawn_loc = carla.Location()
        self._world = None
        self._walker = None
        self._initial_transform = carla.Transform()
        self._world_transform = carla.Transform()
        self._max_spawn_tries = max_spawn_tries

        if world is not None:
            self.bind(world, True)

    def __deepcopy__(self, memo):
        """
        Creates deep copy of the ControlledPedestrian.
        Please note that the result is unbound to world, since it is impossible to spawn
        exact same actor in exactly same location. It is up to copying script to actually
        `.bind()` it as needed.

        :param memo: [description]
        :type memo: [type]
        :return: [description]
        :rtype: ControlledPedestrian
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_spawn_loc':
                setattr(result, k, deepcopy_location(v))
            elif k in ['_initial_transform', '_world_transform']:
                setattr(result, k, deepcopy_transform(v))
            elif k in ['_walker', '_world']:
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def bind(self, world: carla.World, ignore_shift=False):
        """
        Binds the pedestrian to the instance of carla.World

        :param world:
        :type world: carla.World
        """
        # remember current shift from initial position,
        # so we are able to teleport pedestrian there directly
        if not ignore_shift:
            shift = self.transform

        self._world = world
        self._walker = self._spawn_walker()
        self._initial_transform = self._walker.get_transform()
        self._world_transform = self._walker.get_transform()

        if not ignore_shift:  # there was no shift
            self.teleport_by(shift)

        self.apply_pose(True)

    def _spawn_walker(self):
        blueprint_library = self._world.get_blueprint_library()
        matching_blueprints = [bp for bp in blueprint_library.filter("walker.pedestrian.*")
                               if bp.get_attribute('age') == self._age and bp.get_attribute('gender') == self._gender]
        walker_bp = random.choice(matching_blueprints)

        walker = None
        tries = 0
        while walker is None and tries < self._max_spawn_tries:
            tries += 1
            walker_loc = self._world.get_random_location_from_navigation()
            walker = self._world.try_spawn_actor(walker_bp, carla.Transform(walker_loc))

        if walker is None:
            raise RuntimeError("Couldn't spawn walker")
        else:
            self._spawn_loc = walker_loc

        self._world.tick()

        return walker

    def _load_reference_pose(self):
        unreal_pose = load_reference('{}_{}'.format(self._age, self._gender))
        relative_pose = unreal_to_carla(unreal_pose['transforms'])

        return relative_pose

    def teleport_by(self, transform: carla.Transform, cue_tick=False, from_initial=False) -> int:
        """
        Teleports the pedestrian in the world.

        :param transform: Transform relative to the current world transform describing the desired shift
        :type transform: carla.Transform
        :param cue_tick: should carla.World.tick() be called after sending control; defaults to False
        :type cue_tick: bool, optional
        :param from_initial: should teleport be applied from current world position (False, default) or
            from the initial spawn position (True). Mainly used when copying position/movements between 
            pedestrian instances.
        :type from_initial: bool, optional
        :return: World frame number if cue_tick==True else 0
        :rtype: int
        """
        if from_initial:
            reference_transform = self.initial_transform
        else:
            reference_transform = self.world_transform

        self._world_transform = carla.Transform(
            location=carla.Location(
                x=reference_transform.location.x + transform.location.x,
                y=reference_transform.location.y + transform.location.y,
                z=reference_transform.location.z + transform.location.z
            ),
            rotation=carla.Rotation(
                pitch=reference_transform.rotation.pitch + transform.rotation.pitch,
                yaw=reference_transform.rotation.yaw + transform.rotation.yaw,
                roll=reference_transform.rotation.roll + transform.rotation.roll
            )
        )

        if self._walker is not None:
            self._walker.set_transform(self._world_transform)

            if cue_tick:
                return self._world.tick()

        return 0

    def update_pose(self, rotations: Dict[str, carla.Rotation], cue_tick=False) -> int:
        """
        Apply the movement specified as change in local rotations for selected bones.
        For example `pedestrian.update_pose({ 'crl_foreArm__L': carla.Rotation(pitch=60) })`
        will make the arm bend in the elbow by 60deg around Y axis using its current rotation
        plane (which gives roughly 60deg bend around the global Z axis).

        :param rotations: Change in local rotations for selected bones
        :type rotations: Dict[str, carla.Rotation]
        :param cue_tick: should carla.World.tick() be called after sending control; defaults to False
        :type cue_tick: bool, optional
        :return: World frame number if cue_tick==True else 0
        :rtype: int
        """

        self._current_pose.move(rotations)
        return self.apply_pose(cue_tick)

    def apply_pose(self, cue_tick=False, abs_pose_snapshot=None) -> int:
        """
        Applies the current absolute pose to the carla.Walker if it exists.

        :param cue_tick: should carla.World.tick() be called after sending control; defaults to False
        :type cue_tick: bool, optional
        :param abs_pose_snapshot: if not None, will be used instead of self._current_pose.
            This will **NOT** update the internal pose representation.
        :type abs_pose_snapshot: OrderedDict[str, carla.Transform], optional
        :return: World frame number if cue_tick==True else 0
        :rtype: int
        """
        if self._walker is not None:
            control = carla.WalkerBoneControl()

            if abs_pose_snapshot is None:
                abs_pose_snapshot = self._current_pose.absolute

            control.bone_transforms = list(abs_pose_snapshot.items())

            self._walker.apply_control(control)

            if cue_tick:
                return self._world.tick()
        return 0

    @ property
    def age(self) -> str:
        return self._age

    @ property
    def gender(self) -> str:
        return self._gender

    @ property
    def walker(self) -> carla.Actor:
        return self._walker

    @ property
    def world_transform(self) -> carla.Transform:
        if self._walker is not None:
            # if possible, get it from CARLA
            # don't ask me why Z is is some 0.91m above the actual root sometimes
            return self._walker.get_transform()
        return self._world_transform

    @world_transform.setter
    def world_transform(self, transform: carla.Transform):
        if self._walker is not None:
            self._walker.set_transform(transform)
        self._world_transform = transform

    @ property
    def transform(self) -> carla.Transform:
        """
        Current pedestrian transform relative to the position it was spawned at.
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

    @ property
    def initial_transform(self) -> carla.Transform:
        return deepcopy_transform(self._initial_transform)

    @ property
    def current_pose(self) -> Pose:
        return self._current_pose

    @ property
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

    from pedestrians_video_2_carla.carla_utils.destroy import destroy_client_and_world
    from pedestrians_video_2_carla.carla_utils.setup import *

    client, world = setup_client_and_world()
    pedestrian = ControlledPedestrian(world, 'adult', 'female')

    sensor_dict = OrderedDict()
    camera_queue = Queue()

    sensor_dict['camera_rgb'] = setup_camera(
        world, camera_queue, pedestrian
    )

    ticks = 0
    while ticks < 10:
        w_frame = world.tick()

        try:
            sensor_data = camera_queue.get(True, 1.0)
            sensor_data.save_to_disk(
                '/outputs/carla/{:06d}.png'.format(sensor_data.frame))
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
        pedestrian.update_pose({
            'crl_arm__L': carla.Rotation(yaw=-random.random()*15),
            'crl_foreArm__L': carla.Rotation(pitch=-random.random()*15)
        })

    destroy_client_and_world(client, world, sensor_dict)
