from typing import Union, Any

import torch
from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO
from torch.functional import Tensor


class HipsNeckNormalize(object):
    """
    Normalize each sample so that hips x,y = 0,0 and distance between hips & neck == 1.
    """

    def __init__(self, points: Union[BODY_25, COCO] = BODY_25) -> None:
        self.points = points

    def __call__(self, sample: Tensor, *args: Any, **kwds: Any) -> Any:
        hips = self.__get_hips_point(sample)
        neck = sample[:, self.points.neck__C.value, 0:2]
        dist = torch.linalg.vector_norm(neck - hips, dim=1)

        sample[:, :, 0:2] = (sample[:, :, 0:2] -
                             torch.unsqueeze(hips, 1)) / dist.reshape((-1, 1, 1))

        return sample

    def __get_hips_point(self, sample: Tensor):
        try:
            return sample[:, self.points.hips__C.value, 0:2]
        except AttributeError:
            # since COCO does not have hips point, we're using mean of tights
            return sample[:, [self.points.thigh__L.value, self.points.thigh__R.value]].mean(axis=1)[:, 0:2]
