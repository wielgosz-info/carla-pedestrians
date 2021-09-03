import carla
import cameratransform as ct
import numpy as np
import cv2

from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian


class PoseProjection(object):
    def __init__(self, camera_rgb: carla.Sensor, pedestrian: ControlledPedestrian, *args, **kwargs) -> None:
        super().__init__()

        self._pedestrian = pedestrian
        self._image_size = (
            int(camera_rgb.attributes['image_size_x']),
            int(camera_rgb.attributes['image_size_y'])
        )
        self._camera_ct = self._setup_camera(camera_rgb)

    def _setup_camera(self, camera_rgb: carla.Sensor):
        # basic transform is in UE world coords, axes of which are different
        cam_y_offset = camera_rgb.get_transform().location.x - \
            self._pedestrian.world_transform.location.x

        camera_ct = ct.Camera(
            ct.RectilinearProjection(
                image_width_px=self._image_size[0],
                image_height_px=self._image_size[1],
                view_x_deg=float(camera_rgb.attributes['fov']),
                sensor_width_mm=float(camera_rgb.attributes['lens_x_size'])*1000,
                sensor_height_mm=float(camera_rgb.attributes['lens_y_size'])*1000
            ),
            ct.SpatialOrientation(
                pos_y_m=cam_y_offset,
                elevation_m=self._pedestrian.spawn_shift.z,
                heading_deg=180,
                tilt_deg=90
            )
        )

        return camera_ct

    def current_pose_to_points(self):
        # switch from UE world coords, axes of which are different
        ct_transform = carla.Transform(location=carla.Location(
            x=self._pedestrian.transform.location.y,
            y=self._pedestrian.transform.location.x,
            z=self._pedestrian.transform.location.z
        ), rotation=carla.Rotation(
            yaw=-self._pedestrian.transform.rotation.yaw
        ))

        relativeBones = [
            ct_transform.transform(carla.Location(
                x=-bone.location.x,
                y=bone.location.y,
                z=bone.location.z
            ))
            for bone in self._pedestrian.current_absolute_pose.values()
        ]
        return self._camera_ct.imageFromSpace([
            (bone.x, bone.y, bone.z)
            for bone in relativeBones
        ], hide_backpoints=False)

    def current_pose_to_image(self, frame_no):
        points = self.current_pose_to_points()
        rounded = np.round(points).astype(int)

        img = np.zeros((self._image_size[1], self._image_size[0], 4), np.uint8)
        for point in rounded:
            cv2.circle(img, point, 1, [0, 0, 255, 255], 1)

        cv2.line(img, rounded[0], rounded[1], [255, 0, 0, 255], 1)
        cv2.circle(img, rounded[1], 1, [0, 255, 0, 255], 3)

        cv2.imwrite(
            '/outputs/carla/{:06d}_pose.png'.format(frame_no), img)
