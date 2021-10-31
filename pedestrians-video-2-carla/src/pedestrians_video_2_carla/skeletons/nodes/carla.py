from pedestrians_video_2_carla.skeletons.nodes import Skeleton, register_skeleton


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


register_skeleton('CARLA_SKELETON', CARLA_SKELETON)
