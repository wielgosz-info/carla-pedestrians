#!/usr/bin/env python3
"""
KITTI数据集转换为NeuroSLAM格式
将KITTI Odometry数据集转换为与Town01相同的格式

使用方法:
    python convert_kitti_to_neuroslam.py --sequence 00 --kitti_root ../data/02_KITTI_Dataset

输出格式:
    - 图像: 0001.png, 0002.png, ...
    - ground_truth.txt: frame_id,x,y,z,rx,ry,rz
    - visual_odometry.txt: 视觉里程计轨迹
    - dataset_metadata.txt: 数据集元信息
"""

import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
import sys


def parse_kitti_poses(pose_file):
    """
    解析KITTI位姿文件 (12列: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz)
    返回: N x 7 数组 (x, y, z, qw, qx, qy, qz)
    """
    poses = []
    # 使用 utf-8 编码打开，跨平台兼容
    with open(pose_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) != 12:
                continue
                
            # 构建4x4变换矩阵
            T = np.eye(4)
            T[0, :] = [values[0], values[1], values[2], values[3]]
            T[1, :] = [values[4], values[5], values[6], values[7]]
            T[2, :] = [values[8], values[9], values[10], values[11]]
            
            # 提取位置
            x, y, z = T[0, 3], T[1, 3], T[2, 3]
            
            # 提取旋转矩阵并转换为欧拉角
            R = T[:3, :3]
            rot = Rotation.from_matrix(R)
            euler = rot.as_euler('xyz', degrees=True)  # roll, pitch, yaw
            
            poses.append([x, y, z, euler[0], euler[1], euler[2]])
    
    return np.array(poses)


def convert_kitti_sequence(kitti_root, sequence, output_dir):
    """
    转换单个KITTI序列到NeuroSLAM格式
    
    参数:
        kitti_root: KITTI数据集根目录
        sequence: 序列编号 (00, 05, 06等)
        output_dir: 输出目录
    """
    kitti_root = Path(kitti_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"========================================")
    print(f"  转换 KITTI 序列 {sequence}")
    print(f"========================================\n")
    
    # ====================== 核心修改：跨平台相对路径 ======================
    image_dir = kitti_root / "data_odometry_gray" / "sequences" / sequence / "image_0"
    pose_file = kitti_root / "data_odometry_poses" / "poses" / f"{sequence}.txt"
    calib_file = kitti_root / "data_odometry_calib" / "sequences" / f"{sequence}.txt"
    
    # 检查文件存在性
    if not image_dir.exists():
        print(f"❌ 图像目录不存在: {image_dir}")
        return False
    if not pose_file.exists():
        print(f"❌ 位姿文件不存在: {pose_file}")
        return False
    
    print(f"✓ 图像目录: {image_dir}")
    print(f"✓ 位姿文件: {pose_file}")
    
    # 读取位姿数据
    print("\n[1/4] 读取位姿数据...")
    poses = parse_kitti_poses(pose_file)
    print(f"  共 {len(poses)} 帧位姿")
    
    # 获取图像列表
    print("\n[2/4] 读取图像...")
    image_files = sorted(list(image_dir.glob("*.png")))
    print(f"  共 {len(image_files)} 张图像")
    
    # 确保位姿数量与图像数量一致
    num_frames = min(len(poses), len(image_files))
    if len(poses) != len(image_files):
        print(f"  ⚠️  位姿数量({len(poses)}) 与图像数量({len(image_files)}) 不一致")
        print(f"  将使用前 {num_frames} 帧")
    
    # 转换图像
    print(f"\n[3/4] 转换图像 (调整到 120x160)...")
    for i in range(num_frames):
        img_path = str(image_files[i])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ❌ 无法读取图像: {img_path}")
            continue
        
        # 调整图像尺寸为 120x160 (与Town01一致)
        img_resized = cv2.resize(img, (160, 120))
        
        # 保存为4位数编号格式
        output_path = output_dir / f"{i+1:04d}.png"
        cv2.imwrite(str(output_path), img_resized)
        
        if (i + 1) % 500 == 0:
            print(f"  处理进度: {i+1}/{num_frames}")
    
    print(f"  ✓ 图像转换完成")
    
    # 生成ground_truth.txt
    print("\n[4/4] 生成 ground_truth.txt...")
    gt_file = output_dir / "ground_truth.txt"
    with open(gt_file, 'w', encoding='utf-8', newline='') as f:
        # 写入表头
        f.write("frame_id,x,y,z,roll,pitch,yaw\n")
        for i in range(num_frames):
            frame_id = i + 1
            x, y, z, rx, ry, rz = poses[i]
            f.write(f"{frame_id},{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}\n")
    
    print(f"  ✓ Ground Truth 已保存: {gt_file}")
    
    # 生成dataset_metadata.txt
    metadata_file = output_dir / "dataset_metadata.txt"
    trajectory_length = np.sum(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1))
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: KITTI Odometry Sequence {sequence}\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Image Size: 120x160\n")
        f.write(f"Trajectory Length: {trajectory_length:.2f} meters\n")
        f.write(f"Original Size: Variable (adjusted from KITTI)\n")
    
    print(f"  ✓ 元数据已保存: {metadata_file}")
    
    # 计算统计信息
    print(f"\n========================================")
    print(f"  转换完成统计")
    print(f"========================================")
    print(f"  序列编号: {sequence}")
    print(f"  帧数: {num_frames}")
    print(f"  轨迹长度: {trajectory_length:.2f} 米")
    print(f"  平均速度: {trajectory_length / num_frames * 10:.2f} m/s (假设10Hz)")
    print(f"  输出目录: {output_dir}")
    print(f"")
    
    return True


def create_slam_results_dir(output_dir):
    """创建SLAM结果目录"""
    slam_dir = Path(output_dir) / "slam_results"
    slam_dir.mkdir(exist_ok=True)
    print(f"  ✓ 已创建 slam_results 目录")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换KITTI数据集到NeuroSLAM格式")
    parser.add_argument("--sequence", type=str, default="00",
                        help="KITTI序列编号 (00, 05, 06等)")
    parser.add_argument("--kitti_root", type=str, required=True,
                        help="KITTI数据集根目录")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: ../data/01_NeuroSLAM_Datasets/KITTI_Seq_XX)")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.resolve()

    if args.output_dir is None:
        args.output_dir = script_dir / ".." / "data" / "01_NeuroSLAM_Datasets" / f"KITTI_Seq_{args.sequence}"

    # 转换数据集
    success = convert_kitti_sequence(args.kitti_root, args.sequence, args.output_dir)
    
    if success:
        # 创建SLAM结果目录
        create_slam_results_dir(args.output_dir)
        
        print(f"\n{'='*50}")
        print(f"  🎉 数据集转换成功!")
        print(f"{'='*50}")
        print(f"\n现在可以运行测试:")
        print(f"  1. cd ../07_test/test_imu_visual_slam/quickstart")
        print(f"  2. 修改数据集名称为: KITTI_Seq_{args.sequence}")
        print(f"  3. 运行MATLAB脚本")
        print(f"")
    else:
        print(f"\n❌ 数据集转换失败")
        sys.exit(1)
