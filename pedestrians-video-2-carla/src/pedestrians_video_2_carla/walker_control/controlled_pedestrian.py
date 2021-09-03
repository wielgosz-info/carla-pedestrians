from collections import OrderedDict
import copy
from hashlib import new
import random
from typing import Dict
import carla
from scipy.spatial.transform import rotation

from pedestrians_video_2_carla.walker_control.io import load_reference
from pedestrians_video_2_carla.walker_control.transforms import mul_rotations, relative_to_absolute_pose, unreal_to_carla


class ControlledPedestrian(object):
    def __init__(self, world: carla.World, age: str, gender: str, *args, **kwargs):
        super().__init__()

        self._world = world
        self._age = age
        self._gender = gender

        # spawn point (may be different than actual location the pedesrian has spawned, especially Z-wise)
        self._spawn_loc: carla.Location = None
        self._walker = self._spawn_walker()
        # initial world location of where the pedestrian has spawned
        self._initial_transform = self._walker.get_transform()

        # ensure we will always get bones in the same order
        self._current_pose, self._structure = self._structure_to_pose()
        self._current_pose.update(self._apply_reference_pose())

    def _structure_to_pose(self) -> OrderedDict:
        pose = OrderedDict()
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

        absolute_pose = relative_to_absolute_pose(relative_pose, self._structure)

        control = carla.WalkerBoneControl()
        control.bone_transforms = list(absolute_pose.items())

        self._walker.apply_control(control)
        self._world.tick()

        return relative_pose

    def teleport_by(self, transform: carla.Transform, cue_tick=False):
        world_transform = self.world_transform
        self._walker.set_transform(
            carla.Transform(
                location=carla.Location(
                    x=world_transform.location.x + transform.location.x,
                    y=world_transform.location.y + transform.location.y,
                    z=world_transform.location.z + transform.location.z
                ),
                rotation=carla.Rotation(
                    pitch=world_transform.rotation.pitch + transform.rotation.pitch,
                    yaw=world_transform.rotation.yaw + transform.rotation.yaw,
                    roll=world_transform.rotation.roll + transform.rotation.roll
                )
            )
        )

        if cue_tick:
            self._world.tick()

    def apply_movement(self, rotations: Dict[str, carla.Rotation], cue_tick=False):
        """
        Apply the movement specified as change in local rotations for selected bones.
        For example `pedestrian.apply_movement({ 'crl_foreArm__L': carla.Rotation(pitch=60) })`
        will make the arm bend in the elbow by 60deg around Y axis using its current rotation
        plane (which gives roughly 60deg bend around the global Z axis).
        """

        # use getter to ensure we have a copy of self._current_pose
        new_pose = self.current_relative_pose

        # for each defined rotation, we merge it with the current one
        for bone_name, rotation_change in rotations.items():
            new_pose[bone_name].rotation = mul_rotations(
                new_pose[bone_name].rotation, rotation_change)

        absolute_pose = relative_to_absolute_pose(new_pose, self._structure)

        control = carla.WalkerBoneControl()
        control.bone_transforms = list(absolute_pose.items())

        self._walker.apply_control(control)

        if cue_tick:
            self._world.tick()

        self._current_pose = new_pose

    @property
    def age(self) -> str:
        return self._age

    @property
    def gender(self) -> str:
        return self._gender

    @property
    def world_transform(self) -> carla.Transform:
        return self._walker.get_transform()

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
    def current_absolute_pose(self) -> OrderedDict:
        return relative_to_absolute_pose(self._current_pose, self._structure)

    @property
    def current_relative_pose(self) -> OrderedDict:
        # carla.Transform cannot be deepcopied, so we do it manually
        pose_copy = OrderedDict()
        for bone_name, transform in self._current_pose.items():
            pose_copy[bone_name] = carla.Transform(
                location=carla.Location(
                    x=transform.location.x,
                    y=transform.location.y,
                    z=transform.location.z,
                ),
                rotation=carla.Rotation(
                    pitch=transform.rotation.pitch,
                    yaw=transform.rotation.yaw,
                    roll=transform.rotation.roll,
                )
            )
        return pose_copy

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
