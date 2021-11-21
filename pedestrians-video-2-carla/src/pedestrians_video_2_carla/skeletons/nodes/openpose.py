from typing import Type, Union

from torch.functional import Tensor
from pedestrians_video_2_carla.skeletons.nodes import register_skeleton, Skeleton
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor


class BODY_25_SKELETON(Skeleton):
    head__C = 0
    neck__C = 1
    arm__R = 2
    foreArm__R = 3
    hand__R = 4
    arm__L = 5
    foreArm__L = 6
    hand__L = 7
    hips__C = 8
    thigh__R = 9
    leg__R = 10
    foot__R = 11
    thigh__L = 12
    leg__L = 13
    foot__L = 14
    eye__R = 15
    eye__L = 16
    ear__R = 17
    ear__L = 18
    toe__L = 19
    toeEnd__L = 20
    heel__L = 21
    toe__R = 22
    toeEnd__R = 23
    heel__R = 24

    @classmethod
    def get_extractor(cls) -> Type[HipsNeckExtractor]:
        return OpenPoseHipsNeckExtractor(cls)


class COCO_SKELETON(Skeleton):
    head__C = 0
    neck__C = 1
    arm__R = 2
    foreArm__R = 3
    hand__R = 4
    arm__L = 5
    foreArm__L = 6
    hand__L = 7
    thigh__R = 8
    leg__R = 9
    foot__R = 10
    thigh__L = 11
    leg__L = 12
    foot__L = 13
    eye__R = 14
    eye__L = 15
    ear__R = 16
    ear__L = 17

    @classmethod
    def get_extractor(cls) -> Type[HipsNeckExtractor]:
        return OpenPoseHipsNeckExtractor(cls)


class OpenPoseHipsNeckExtractor(HipsNeckExtractor):
    def __init__(self, input_nodes: Type[Union[BODY_25_SKELETON, COCO_SKELETON]] = BODY_25_SKELETON) -> None:
        super().__init__(input_nodes)

    def get_hips_point(self, sample: Tensor) -> Tensor:
        try:
            return sample[..., self.input_nodes.hips__C.value, :]
        except AttributeError:
            # since COCO does not have hips point, we're using mean of tights
            return sample[..., [self.input_nodes.thigh__L.value, self.input_nodes.thigh__R.value], :].mean(axis=-2)

    def get_neck_point(self, sample: Tensor) -> Tensor:
        return sample[..., self.input_nodes.neck__C.value, :]


register_skeleton('BODY_25_SKELETON', BODY_25_SKELETON, [
    (CARLA_SKELETON.crl_hips__C, BODY_25_SKELETON.hips__C),
    (CARLA_SKELETON.crl_arm__L, BODY_25_SKELETON.arm__L),
    (CARLA_SKELETON.crl_foreArm__L, BODY_25_SKELETON.foreArm__L),
    (CARLA_SKELETON.crl_hand__L, BODY_25_SKELETON.hand__L),
    (CARLA_SKELETON.crl_neck__C, BODY_25_SKELETON.neck__C),
    (CARLA_SKELETON.crl_Head__C, BODY_25_SKELETON.head__C),
    (CARLA_SKELETON.crl_arm__R, BODY_25_SKELETON.arm__R),
    (CARLA_SKELETON.crl_foreArm__R, BODY_25_SKELETON.foreArm__R),
    (CARLA_SKELETON.crl_hand__R, BODY_25_SKELETON.hand__R),
    (CARLA_SKELETON.crl_eye__L, BODY_25_SKELETON.eye__L),
    (CARLA_SKELETON.crl_eye__R, BODY_25_SKELETON.eye__R),
    (CARLA_SKELETON.crl_thigh__R, BODY_25_SKELETON.thigh__R),
    (CARLA_SKELETON.crl_leg__R, BODY_25_SKELETON.leg__R),
    (CARLA_SKELETON.crl_foot__R, BODY_25_SKELETON.foot__R),
    (CARLA_SKELETON.crl_toe__R, BODY_25_SKELETON.toe__R),
    (CARLA_SKELETON.crl_toeEnd__R, BODY_25_SKELETON.toeEnd__R),
    (CARLA_SKELETON.crl_thigh__L, BODY_25_SKELETON.thigh__L),
    (CARLA_SKELETON.crl_leg__L, BODY_25_SKELETON.leg__L),
    (CARLA_SKELETON.crl_foot__L, BODY_25_SKELETON.foot__L),
    (CARLA_SKELETON.crl_toe__L, BODY_25_SKELETON.toe__L),
    (CARLA_SKELETON.crl_toeEnd__L, BODY_25_SKELETON.toeEnd__L),
])

register_skeleton('COCO_SKELETON', COCO_SKELETON, [
    (CARLA_SKELETON.crl_arm__L, COCO_SKELETON.arm__L),
    (CARLA_SKELETON.crl_foreArm__L, COCO_SKELETON.foreArm__L),
    (CARLA_SKELETON.crl_hand__L, COCO_SKELETON.hand__L),
    (CARLA_SKELETON.crl_neck__C, COCO_SKELETON.neck__C),
    (CARLA_SKELETON.crl_Head__C, COCO_SKELETON.head__C),
    (CARLA_SKELETON.crl_arm__R, COCO_SKELETON.arm__R),
    (CARLA_SKELETON.crl_foreArm__R, COCO_SKELETON.foreArm__R),
    (CARLA_SKELETON.crl_hand__R, COCO_SKELETON.hand__R),
    (CARLA_SKELETON.crl_eye__L, COCO_SKELETON.eye__L),
    (CARLA_SKELETON.crl_eye__R, COCO_SKELETON.eye__R),
    (CARLA_SKELETON.crl_thigh__R, COCO_SKELETON.thigh__R),
    (CARLA_SKELETON.crl_leg__R, COCO_SKELETON.leg__R),
    (CARLA_SKELETON.crl_foot__R, COCO_SKELETON.foot__R),
    (CARLA_SKELETON.crl_thigh__L, COCO_SKELETON.thigh__L),
    (CARLA_SKELETON.crl_leg__L, COCO_SKELETON.leg__L),
    (CARLA_SKELETON.crl_foot__L, COCO_SKELETON.foot__L),
])
