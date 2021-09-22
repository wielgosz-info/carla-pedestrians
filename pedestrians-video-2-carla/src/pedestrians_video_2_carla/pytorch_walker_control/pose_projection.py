import carla
import numpy as np
import torch
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.pose_projection import \
    PoseProjection
from pytorch3d.renderer.cameras import (PerspectiveCameras,
                                        look_at_view_transform)
from torch.functional import Tensor
from torch.types import Device


class P3dPoseProjection(PoseProjection, torch.nn.Module):
    def __init__(self, device: Device, pedestrian: ControlledPedestrian, camera_rgb: carla.Sensor = None, *args, **kwargs):
        self._device = device

        super().__init__(pedestrian=pedestrian, camera_rgb=camera_rgb, *args, **kwargs)

    def _setup_camera(self, camera_rgb: carla.Sensor):
        # basic transform is in UE world coords, axes of which are different
        # additionally, we need to correct spawn shift error
        cam_z_offset = camera_rgb.get_transform().location.x - \
            self._pedestrian.world_transform.location.x + \
            self._pedestrian.spawn_shift.x
        cam_y_offset = camera_rgb.get_transform().location.z - \
            self._pedestrian.world_transform.location.z + \
            self._pedestrian.spawn_shift.z

        R, T = look_at_view_transform(
            eye=((0, cam_y_offset, cam_z_offset),), at=((0, cam_y_offset, 0), ))

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
        # switch from UE world coords, axes of which are different
        root_transform = carla.Transform(location=carla.Location(
            x=-self._pedestrian.transform.location.y,
            y=self._pedestrian.transform.location.x,
            z=self._pedestrian.transform.location.z
        ), rotation=carla.Rotation(
            yaw=self._pedestrian.transform.rotation.yaw
        ))

        bones = [[t.x, t.z, t.y] for t in [
            root_transform.transform(bone.location)
            for bone in self._pedestrian.current_pose.absolute.values()
        ]]

        return self.forward(torch.Tensor(bones).to(self._device))

    def _raw_to_pixel_points(self, points):
        return np.round(points.cpu().numpy()[..., :2]).astype(int)

    def forward(self, x: Tensor):
        """
        Projects 3D points to 2D using predefined camera.

        TODO: This method uses PyTorch3D coordinates system (right-handed, negative Z
            and radians (-roll, -pitch, -yaw) angles when compared to CARLA).

        :param x: (..., 3)
        :type x: torch.Tensor
        :return: Points projected to 2D. Returned tensor has the same shape as input one: (..., 3)
        :rtype: torch.Tensor
        """
        return self._camera.transform_points_screen(x)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    pedestrian = ControlledPedestrian(None, 'adult', 'female')
    p3d_projection = P3dPoseProjection(device, pedestrian, None)
    p3d_points = p3d_projection.current_pose_to_points()
    p3d_projection.current_pose_to_image('reference_pytorch3d', p3d_points)
