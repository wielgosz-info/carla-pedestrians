#!/usr/bin/env python3
"""
KITTI EKF融合脚本
基于Town的IMU_Vision_Fusion_EKF.py，用于离线处理KITTI数据
生成带有真实EKF融合的fusion_pose.txt
"""

import os
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from visual_odometry_opencv import VisualOdometry, ScaleEstimator

# 导入独立EKF (不依赖CARLA)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ekf_standalone import EKF_VIO

# ======================== 配置参数 ========================
KITTI_SEQUENCE = '07'
# 使用相对路径（相对于当前脚本位置）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KITTI_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', f'KITTI_{KITTI_SEQUENCE}')
KITTI_DATA_PATH = os.path.abspath(KITTI_DATA_PATH)  # 规范化路径
OUTPUT_DIR = KITTI_DATA_PATH

# 相机参数 (KITTI Sequence 07, Camera 0)
CAMERA_PARAMS = {
    'fx': 718.856,    # focal length x
    'fy': 718.856,    # focal length y
    'cx': 607.1928,   # principal point x
    'cy': 185.2157,   # principal point y
}

# EKF参数 (dt约为0.1秒，KITTI约10Hz)
EKF_DT = 0.1

# 视觉里程计参数
VO_PARAMS = {
    'focal_length': CAMERA_PARAMS['fx'],
    'principal_point': (CAMERA_PARAMS['cx'], CAMERA_PARAMS['cy']),
    'min_features': 2000,
    'max_features': 3000
}

# ===========================================================

class KITTIDataLoader:
    """KITTI数据加载器"""
    
    def __init__(self, data_path, sequence):
        self.data_path = data_path
        self.sequence = sequence
        
        # 图像路径
        self.image_path = os.path.join(data_path, 'image_0')
        
        # OXTS路径
        self.oxts_path = os.path.join(data_path, 'oxts', 'data')
        
        # 加载时间戳
        self.load_timestamps()
        
        # 统计帧数
        self.num_frames = len(self.timestamps)
        print(f"✓ 加载KITTI序列 {sequence}: {self.num_frames} 帧")
    
    def load_timestamps(self):
        """加载时间戳"""
        times_file = os.path.join(self.data_path, 'times.txt')
        self.timestamps = []
        with open(times_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.timestamps.append(float(line.strip()))
    
    def get_image(self, idx):
        """读取图像"""
        img_file = os.path.join(self.image_path, f'{idx:06d}.png')
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"警告: 无法读取图像 {img_file}")
        return img
    
    def get_oxts(self, idx):
        """读取OXTS数据 (IMU + GPS)"""
        oxts_file = os.path.join(self.oxts_path, f'{idx:010d}.txt')
        
        with open(oxts_file, 'r', encoding='utf-8') as f:
            data = list(map(float, f.readline().split()))
        
        # 解析OXTS数据
        oxts = {
            'lat': data[0],      # 纬度
            'lon': data[1],      # 经度
            'alt': data[2],      # 高度
            'roll': data[3],     # 横滚角 (rad)
            'pitch': data[4],    # 俯仰角 (rad)
            'yaw': data[5],      # 航向角 (rad)
            'vn': data[6],       # 北向速度
            've': data[7],       # 东向速度
            'vf': data[8],       # 前向速度
            'vl': data[9],       # 左向速度
            'vu': data[10],      # 上向速度
            'ax': data[11],      # x加速度 (车体坐标系)
            'ay': data[12],      # y加速度
            'az': data[13],      # z加速度
            'af': data[14],      # 前向加速度
            'al': data[15],      # 左向加速度
            'au': data[16],      # 上向加速度
            'wx': data[17],      # x角速度 (rad/s)
            'wy': data[18],      # y角速度
            'wz': data[19],      # z角速度
        }
        
        return oxts


def latlon_to_mercator(lat, lon, alt, origin_lat, origin_lon, origin_alt):
    """
    将GPS经纬度转换为局部笛卡尔坐标 (Mercator投影)
    """
    # 地球半径 (米)
    R_earth = 6378137.0
    
    # 转为弧度
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    origin_lat_rad = np.radians(origin_lat)
    origin_lon_rad = np.radians(origin_lon)
    
    # Mercator投影
    x = R_earth * (lon_rad - origin_lon_rad) * np.cos(origin_lat_rad)
    y = R_earth * (lat_rad - origin_lat_rad)
    z = alt - origin_alt
    
    return x, y, z


def main():
    print("\n" + "="*60)
    print("KITTI EKF融合脚本")
    print("="*60 + "\n")
    
    # 1. 加载数据
    print("[1/5] 加载KITTI数据...")
    loader = KITTIDataLoader(KITTI_DATA_PATH, KITTI_SEQUENCE)
    
    # 2. 初始化视觉里程计
    print("[2/5] 初始化视觉里程计...")
    # 构建相机内参矩阵
    camera_matrix = np.array([
        [CAMERA_PARAMS['fx'],  0, CAMERA_PARAMS['cx']],
        [ 0, CAMERA_PARAMS['fy'], CAMERA_PARAMS['cy']],
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    vo = VisualOdometry(camera_matrix=camera_matrix, dist_coeffs=None)
    
    scale_estimator = ScaleEstimator(alpha=0.9, use_fixed_scale=False)
    
    # 3. 初始化EKF
    print("[3/5] 初始化EKF...")
    # 获取初始位置和速度
    oxts_0 = loader.get_oxts(0)
    # 使用相对坐标系（与poses.txt一致）：初始位置和朝向都为0
    init_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x,y,z,roll,pitch,yaw（相对第一帧）
    init_vel = [0.0, 0.0, 0.0]  # 零初始速度（不使用GPS/INS速度，纯IMU+VO融合）
    
    ekf = EKF_VIO(init_pose=init_pose, init_vel=init_vel, dt=EKF_DT)
    
    # 保存初始yaw用于VO（相对坐标系）
    init_yaw = 0.0  # 相对第一帧，初始朝向定义为0
    
    # 4. 设置原点（GPS坐标转换）
    print("[4/5] 设置坐标原点...")
    origin_lat = oxts_0['lat']
    origin_lon = oxts_0['lon']
    origin_alt = oxts_0['alt']
    print(f"  原点GPS: lat={origin_lat:.6f}, lon={origin_lon:.6f}, alt={origin_alt:.2f}m")
    
    # 5. 处理所有帧
    print(f"[5/5] 开始处理 {loader.num_frames} 帧...\n")
    
    # 准备输出文件
    fusion_file = os.path.join(OUTPUT_DIR, 'fusion_pose_ekf.txt')
    vo_file = os.path.join(OUTPUT_DIR, 'visual_odometry_ekf.txt')
    gt_file = os.path.join(OUTPUT_DIR, 'ground_truth_ekf.txt')
    
    with open(fusion_file, 'w', encoding='utf-8') as f_fusion, \
         open(vo_file, 'w', encoding='utf-8') as f_vo, \
         open(gt_file, 'w', encoding='utf-8') as f_gt:
        
        # 写入表头
        f_fusion.write("timestamp,pos_x,pos_y,pos_z,roll,pitch,yaw,imu_pos_x,imu_pos_y,imu_pos_z,vel_x,vel_y,vel_z,uncertainty_x,uncertainty_y,uncertainty_z\n")
        f_vo.write("timestamp,vo_x,vo_y,vo_z,vo_roll,vo_pitch,vo_yaw,num_matches,scale\n")
        f_gt.write("timestamp,pos_x,pos_y,pos_z,roll,pitch,yaw,vel_x,vel_y,vel_z\n")
        
        # 准备输出文件
        prev_img = None
        vo_pos = np.array([0.0, 0.0, 0.0])  # VO累积位置（仅用于记录）
        vo_yaw = init_yaw  # VO累积yaw，从真实初始朝向开始
        imu_pos = np.array([0.0, 0.0, 0.0])  # IMU累积位置（仅用于记录）
        
        for frame_idx in range(loader.num_frames):
            timestamp = loader.timestamps[frame_idx]
            
            # 读取图像
            img = loader.get_image(frame_idx)
            if img is None:
                print(f"跳过帧{frame_idx}: 图像读取失败")
                continue
            
            # 读取OXTS数据
            oxts = loader.get_oxts(frame_idx)
            
            # === IMU预测 ===
            # 创建IMU数据结构 (兼容EKF_VIO接口)
            class IMUData:
                def __init__(self, acc, gyro):
                    self.accelerometer = type('obj', (object,), {
                        'x': acc[0], 'y': acc[1], 'z': acc[2]
                    })
                    self.gyroscope = type('obj', (object,), {
                        'x': gyro[0], 'y': gyro[1], 'z': gyro[2]
                    })
            
            imu_data = IMUData(
                acc=np.array([oxts['ax'], oxts['ay'], oxts['az']]),
                gyro=np.array([oxts['wx'], oxts['wy'], oxts['wz']])
            )
            
            ekf.imu_prediction(imu_data)
            
            # 累积IMU位置（用于记录）
            dt = EKF_DT
            if frame_idx > 0:
                dt = timestamp - loader.timestamps[frame_idx - 1]
            vel = ekf.get_current_velocity()
            imu_pos += vel * dt
            
            # === 视觉里程计 ===
            num_matches = 0
            scale = 1.0
            
            # 第一帧初始化
            if prev_img is None:
                vo.process_frame(img)  # 初始化
                prev_img = img.copy()
                # 第一帧，写入初始数据后继续
            else:
                # 处理当前帧（获取相对运动）
                _, num_matches = vo.process_frame(img)
                
                # 获取VO的累积绝对位姿
                vo_abs_pose = vo.get_absolute_pose()
                
                if vo_abs_pose is not None and num_matches > 10:
                    # vo_abs_pose格式: [x, y, z, roll, pitch, yaw]（相机坐标系，累积位姿）
                    
                    # 相机坐标系到KITTI世界坐标系转换
                    # KITTI相机: X右 Y下 Z前
                    # KITTI世界: X前 Y左 Z上
                    vo_x_cam, vo_y_cam, vo_z_cam = vo_abs_pose[0], vo_abs_pose[1], vo_abs_pose[2]
                    vo_yaw_cam = vo_abs_pose[5]
                    
                    # 坐标转换
                    vo_x_world = vo_z_cam      # 相机Z前 → 世界X前
                    vo_y_world = -vo_x_cam     # -相机X右 → 世界Y左
                    vo_z_world = -vo_y_cam     # -相机Y下 → 世界Z上
                    
                    # Yaw转换：相机yaw（绕相机Z轴=世界X轴）→ 世界yaw（绕世界Z轴）
                    # 相机绕Z轴旋转 = 车辆左右转 = 世界yaw
                    vo_yaw_world = vo_yaw_cam
                    
                    # 归一化yaw
                    vo_yaw_world = np.arctan2(np.sin(vo_yaw_world), np.cos(vo_yaw_world))
                    
                    # 保存VO位姿
                    vo_pos = np.array([vo_x_world, vo_y_world, vo_z_world])
                    vo_yaw = vo_yaw_world
                    
                    # EKF视觉更新
                    visual_measurement = np.array([
                        vo_x_world, vo_y_world, vo_z_world,
                        0.0, 0.0, vo_yaw_world
                    ])
                    ekf.visual_update(visual_measurement)
                
                prev_img = img.copy()
            
            # === 获取融合结果 ===
            fusion_pos, fusion_att = ekf.get_current_pose()
            fusion_vel = ekf.get_current_velocity()
            pos_uncertainty = ekf.get_position_uncertainty()
            
            # 归一化fusion yaw到[-π, π]
            fusion_att[2] = np.arctan2(np.sin(fusion_att[2]), np.cos(fusion_att[2]))
            
            # === Ground Truth (从OXTS转换) ===
            gt_x, gt_y, gt_z = latlon_to_mercator(
                oxts['lat'], oxts['lon'], oxts['alt'],
                origin_lat, origin_lon, origin_alt
            )
            gt_pos = np.array([gt_x, gt_y, gt_z])
            gt_vel = np.array([oxts['vf'], oxts['vl'], oxts['vu']])
            gt_att = np.array([oxts['roll'], oxts['pitch'], oxts['yaw']])
            
            # === 写入数据 ===
            # Fusion pose (姿态用弧度)
            f_fusion.write(f"{timestamp:.6f},"
                          f"{fusion_pos[0]:.6f},{fusion_pos[1]:.6f},{fusion_pos[2]:.6f},"
                          f"{fusion_att[0]:.6f},{fusion_att[1]:.6f},{fusion_att[2]:.6f},"
                          f"{imu_pos[0]:.6f},{imu_pos[1]:.6f},{imu_pos[2]:.6f},"
                          f"{fusion_vel[0]:.6f},{fusion_vel[1]:.6f},{fusion_vel[2]:.6f},"
                          f"{pos_uncertainty[0]:.6f},{pos_uncertainty[1]:.6f},{pos_uncertainty[2]:.6f}\n")
            
            # Visual odometry
            f_vo.write(f"{timestamp:.6f},"
                      f"{vo_pos[0]:.6f},{vo_pos[1]:.6f},{vo_pos[2]:.6f},"
                      f"0.0,0.0,{np.degrees(vo_yaw):.6f},"  # roll=pitch=0, 只有yaw
                      f"{num_matches},{scale:.4f}\n")
            
            # Ground truth
            f_gt.write(f"{timestamp:.6f},"
                      f"{gt_pos[0]:.6f},{gt_pos[1]:.6f},{gt_pos[2]:.6f},"
                      f"{np.degrees(gt_att[0]):.6f},{np.degrees(gt_att[1]):.6f},{np.degrees(gt_att[2]):.6f},"
                      f"{gt_vel[0]:.6f},{gt_vel[1]:.6f},{gt_vel[2]:.6f}\n")
            
            # 进度显示
            if (frame_idx + 1) % 100 == 0 or frame_idx == 0:
                progress = (frame_idx + 1) / loader.num_frames * 100
                print(f"  进度: {frame_idx+1}/{loader.num_frames} ({progress:.1f}%) | "
                      f"VT匹配: {num_matches} | 尺度: {scale:.3f}")
    
    print(f"\n✅ EKF融合完成！")
    print(f"  融合数据: {fusion_file}")
    print(f"  视觉里程计: {vo_file}")
    print(f"  Ground Truth: {gt_file}")
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
