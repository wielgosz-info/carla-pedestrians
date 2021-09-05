from typing import Tuple
import numpy as np

from pedestrians_video_2_carla.utils.openpose import alternative_hips_neck


def scale_projection_by_height(points: np.ndarray, hips: Tuple[float, float]):
    height = points[:, 1].max() - points[:, 1].min()
    zeros = np.array([
        hips[0],  # X of hips point
        points[:, 1].min()
    ])
    points = (points - zeros) / height

    return points


def scale_projections_by_height(image_projection_points, image_openpose_points, empty_pose):
    # we need to substitute some points from the original projection
    alt_projection_points, (hips_idx, _) = alternative_hips_neck(
        image_projection_points, empty_pose)

    # not every openpose point can be mapped to carla skeleton, so we need to skip some
    not_NaNs = ~np.isnan(image_openpose_points).any(axis=1)

    projection_points = scale_projection_by_height(
        alt_projection_points[not_NaNs], alt_projection_points[hips_idx])
    openpose_points = scale_projection_by_height(
        image_openpose_points[not_NaNs], image_openpose_points[hips_idx])

    return projection_points, openpose_points
