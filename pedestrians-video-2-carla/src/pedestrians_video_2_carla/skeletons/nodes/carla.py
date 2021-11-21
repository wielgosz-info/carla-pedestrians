from typing import Type

from torch.functional import Tensor
from pedestrians_video_2_carla.skeletons.nodes import Skeleton, register_skeleton
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor


class CARLA_SKELETON(Skeleton):
    crl_root = 0
    crl_hips__C = 1
    crl_spine__C = 2
    crl_spine01__C = 3
    crl_shoulder__L = 4
    crl_arm__L = 5
    crl_foreArm__L = 6
    crl_hand__L = 7
    crl_neck__C = 8
    crl_Head__C = 9
    crl_eye__L = 10
    crl_eye__R = 11
    crl_shoulder__R = 12
    crl_arm__R = 13
    crl_foreArm__R = 14
    crl_hand__R = 15
    crl_thigh__R = 16
    crl_leg__R = 17
    crl_foot__R = 18
    crl_toe__R = 19
    crl_toeEnd__R = 20
    crl_thigh__L = 21
    crl_leg__L = 22
    crl_foot__L = 23
    crl_toe__L = 24
    crl_toeEnd__L = 25

    @classmethod
    def get_extractor(cls) -> Type[HipsNeckExtractor]:
        return CarlaHipsNeckExtractor(cls)


class CarlaHipsNeckExtractor(HipsNeckExtractor):
    def __init__(self, input_nodes: Type[CARLA_SKELETON] = CARLA_SKELETON) -> None:
        super().__init__(input_nodes)

    def get_hips_point(self, sample: Tensor) -> Tensor:
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between tights as a reference instead
        return sample[..., [self.input_nodes.crl_thigh__L.value, self.input_nodes.crl_thigh__R.value], :].mean(axis=-2)

    def get_neck_point(self, sample: Tensor) -> Tensor:
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between shoulders as a reference instead
        return sample[..., [self.input_nodes.crl_shoulder__L.value, self.input_nodes.crl_shoulder__R.value], :].mean(axis=-2)


register_skeleton('CARLA_SKELETON', CARLA_SKELETON)
