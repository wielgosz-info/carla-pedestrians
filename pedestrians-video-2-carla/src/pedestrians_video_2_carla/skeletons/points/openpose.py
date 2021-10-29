from enum import Enum

from pedestrians_video_2_carla.skeletons.points import MAPPINGS
from pedestrians_video_2_carla.skeletons.points.carla import CARLA_SKELETON


class BODY_25(Enum):
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


class COCO(Enum):
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


# TODO: add weights?
MAPPINGS.update({
    BODY_25: [
        (CARLA_SKELETON.crl_hips__C, BODY_25.hips__C),
        (CARLA_SKELETON.crl_arm__L, BODY_25.arm__L),
        (CARLA_SKELETON.crl_foreArm__L, BODY_25.foreArm__L),
        (CARLA_SKELETON.crl_hand__L, BODY_25.hand__L),
        (CARLA_SKELETON.crl_neck__C, BODY_25.neck__C),
        (CARLA_SKELETON.crl_Head__C, BODY_25.head__C),
        (CARLA_SKELETON.crl_arm__R, BODY_25.arm__R),
        (CARLA_SKELETON.crl_foreArm__R, BODY_25.foreArm__R),
        (CARLA_SKELETON.crl_hand__R, BODY_25.hand__R),
        (CARLA_SKELETON.crl_eye__L, BODY_25.eye__L),
        (CARLA_SKELETON.crl_eye__R, BODY_25.eye__R),
        (CARLA_SKELETON.crl_thigh__R, BODY_25.thigh__R),
        (CARLA_SKELETON.crl_leg__R, BODY_25.leg__R),
        (CARLA_SKELETON.crl_foot__R, BODY_25.foot__R),
        (CARLA_SKELETON.crl_toe__R, BODY_25.toe__R),
        (CARLA_SKELETON.crl_toeEnd__R, BODY_25.toeEnd__R),
        (CARLA_SKELETON.crl_thigh__L, BODY_25.thigh__L),
        (CARLA_SKELETON.crl_leg__L, BODY_25.leg__L),
        (CARLA_SKELETON.crl_foot__L, BODY_25.foot__L),
        (CARLA_SKELETON.crl_toe__L, BODY_25.toe__L),
        (CARLA_SKELETON.crl_toeEnd__L, BODY_25.toeEnd__L),
    ],
    COCO: [
        (CARLA_SKELETON.crl_arm__L, COCO.arm__L),
        (CARLA_SKELETON.crl_foreArm__L, COCO.foreArm__L),
        (CARLA_SKELETON.crl_hand__L, COCO.hand__L),
        (CARLA_SKELETON.crl_neck__C, COCO.neck__C),
        (CARLA_SKELETON.crl_Head__C, COCO.head__C),
        (CARLA_SKELETON.crl_arm__R, COCO.arm__R),
        (CARLA_SKELETON.crl_foreArm__R, COCO.foreArm__R),
        (CARLA_SKELETON.crl_hand__R, COCO.hand__R),
        (CARLA_SKELETON.crl_eye__L, COCO.eye__L),
        (CARLA_SKELETON.crl_eye__R, COCO.eye__R),
        (CARLA_SKELETON.crl_thigh__R, COCO.thigh__R),
        (CARLA_SKELETON.crl_leg__R, COCO.leg__R),
        (CARLA_SKELETON.crl_foot__R, COCO.foot__R),
        (CARLA_SKELETON.crl_thigh__L, COCO.thigh__L),
        (CARLA_SKELETON.crl_leg__L, COCO.leg__L),
        (CARLA_SKELETON.crl_foot__L, COCO.foot__L),
    ]
})
