import carla
import numpy as np
import torch
from torch._C import device
from pedestrians_video_2_carla.pytorch_walker_control.pose import P3dPose
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.pose_projection import \
    PoseProjection
from pytorch3d.renderer.cameras import (PerspectiveCameras,
                                        look_at_view_transform)
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.transforms.transform3d import Rotate, Translate
from torch.functional import Tensor
from torch.types import Device


class P3dPoseProjection(PoseProjection, torch.nn.Module):
    def __init__(self, device: Device, pedestrian: ControlledPedestrian, camera_rgb: carla.Sensor = None, *args, **kwargs):
        self._device = device

        super().__init__(pedestrian=pedestrian, camera_rgb=camera_rgb, *args, **kwargs)

    def _setup_camera(self, camera_rgb: carla.Sensor):
        # basic transform is in UE world coords, axes of which are different
        # additionally, we need to correct spawn shift error
        distance = camera_rgb.get_transform().location.x - \
            self._pedestrian.world_transform.location.x + \
            self._pedestrian.spawn_shift.x
        elevation = camera_rgb.get_transform().location.z - \
            self._pedestrian.world_transform.location.z + \
            self._pedestrian.spawn_shift.z

        R, T = look_at_view_transform(
            eye=((distance, 0, -elevation),), at=((0, 0, -elevation),), up=((0, 0, -1),))

        # from CameraTransform docs/code we get:
        # focallength_px_x = (focallength_mm_x / sensor_width_mm) * image_width_px
        # and
        # focallength_px_x = image_width_px / (2 * np.tan(np.deg2rad(view_x) / 2))
        # so
        # focallength_mm_x = sensor_width_mm / (2 * np.tan(np.deg2rad(view_x) / 2))
        view_x_deg = float(camera_rgb.attributes['fov']),
        sensor_width_mm = float(camera_rgb.attributes['lens_x_size'])*1000
        focal_length_mm = sensor_width_mm / (2 * np.tan(np.deg2rad(view_x_deg) / 2))

        # We cannot use the FOVPerspectiveCameras due to existence of sensor width
        cameras = PerspectiveCameras(
            device=self._device,
            in_ndc=False,
            focal_length=focal_length_mm*10,
            principal_point=((self._image_size[0]/2, self._image_size[1]/2),),
            image_size=((self._image_size[1], self._image_size[0]),),
            R=R, T=T
        )

        return cameras

    def current_pose_to_points(self):
        root_transform = self._pedestrian.transform
        bones_len = len(self._pedestrian.current_pose.empty)

        if hasattr(self._pedestrian.current_pose, 'forward'):
            absolute = self._pedestrian.current_pose.forward(
                torch.zeros((bones_len, 3), device=self._device))
        else:
            absolute = torch.tensor(
                [(t.location.x, t.location.y, -t.location.z)
                 for t in self._pedestrian.current_pose.absolute.values()],
                device=self._device, dtype=torch.float32)

        loc = torch.tensor(((
            root_transform.location.x,
            root_transform.location.y,
            -root_transform.location.z
        ), ), device=self._device, dtype=torch.float32)

        rot = torch.tensor((
            np.deg2rad(-root_transform.rotation.roll),
            np.deg2rad(-root_transform.rotation.pitch),
            np.deg2rad(-root_transform.rotation.yaw)
        ), device=self._device, dtype=torch.float32)

        p3d_points = self.forward(absolute, loc, rot)
        return p3d_points.cpu().numpy()[..., :2]

    def forward(self, x: Tensor, loc: Tensor, rot: Tensor):
        """
        Projects 3D points to 2D using predefined camera.

        This method uses PyTorch3D coordinates system (right-handed, negative Z
            and radians (-roll, -pitch, -yaw) angles when compared to CARLA).

        :param x: (..., (x, y, -z)) Tensor containing absolute pose values (as outputted by P3dPose.forward)
        :type x: torch.Tensor
        :param loc: (3,) Tensor containing pedestrian relative world transform (x, y, -z)
        :type loc: torch.Tensor
        :param rot: (3,) Tensor containing pedestrian relative world rotation in radians (-roll, -pitch, -yaw)
        :type rot: torch.Tensor
        :return: Points projected to 2D. Returned tensor has the same shape as input one: (..., 3),
            but only [..., :2] are usable.
        :rtype: torch.Tensor
        """
        p3d_2_world = torch.tensor((
            (0., -1., 0.),
            (1., 0., 0.),
            (0., 0., 1.)
        ), device=rot.device)

        world_rot = euler_angles_to_matrix(rot, "XYZ")
        world_loc = loc
        world_x = torch.mm(x, p3d_2_world)

        world_pos = Rotate(world_rot).compose(
            Translate(world_loc)).transform_points(world_x)
        projected_x = self._camera.transform_points_screen(world_pos)
        return projected_x


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    pedestrian = ControlledPedestrian(
        None, 'adult', 'female', pose_cls=P3dPose, device=device)
    p3d_projection = P3dPoseProjection(device, pedestrian, None)
    p3d_points = p3d_projection.current_pose_to_points()
    p3d_projection.current_pose_to_image('reference_pytorch3d', p3d_points)
