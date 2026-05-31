"""
真正的视觉里程计实现
使用OpenCV从RGB图像序列中估计相机运动
"""
import numpy as np
import cv2

class VisualOdometry:
    """基于特征的单目视觉里程计"""
    
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """
        初始化视觉里程计
        
        参数:
            camera_matrix: 相机内参矩阵 [3x3]
            dist_coeffs: 畸变系数
        """
        # CARLA默认相机参数 (640x480, FOV=90度)
        if camera_matrix is None:
            fx = fy = 640 / (2 * np.tan(np.radians(90) / 2))
            cx, cy = 640 / 2, 480 / 2
            self.K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ], dtype=np.float32)
        else:
            self.K = camera_matrix
        
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
        # 特征检测器（使用ORB - 快速且免费）
        self.detector = cv2.ORB_create(nfeatures=2000)

        # 特征匹配器
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # 上一帧数据
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None

        # 累积位姿
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))

        # 统计信息
        self.num_inliers = 0
        self.scale = 1.0  # 单目VO无法恢复真实尺度
        self.scale_history = []  # 尺度历史（用于滤波）
        self.max_scale_history = 10  # 保持最近10帧

        # 预计算相机内参逆矩阵，加速后续计算
        self.K_inv = np.linalg.inv(self.K)
        
    def process_frame(self, frame):
        """
        处理新的一帧图像
        
        参数:
            frame: RGB图像 [H, W, 3]
        
        返回:
            relative_pose: 相对位姿 [x, y, z, roll, pitch, yaw] 或 None（首帧）
            num_matches: 特征匹配数量
        """
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # 检测特征
        kp, des = self.detector.detectAndCompute(gray, None)
        
        # 首帧初始化
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return None, 0
        
        # 特征匹配
        if des is None or self.prev_des is None or len(des) < 10 or len(self.prev_des) < 10:
            # 特征不足，返回零运动
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return [0, 0, 0, 0, 0, 0], 0
        
        matches = self.matcher.knnMatch(self.prev_des, des, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            # 匹配点太少
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return [0, 0, 0, 0, 0, 0], 0
        
        # 提取匹配点坐标
        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good_matches])
        
        # 计算本质矩阵
        E, mask = cv2.findEssentialMat(
            pts_prev, pts_curr, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None or mask is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return [0, 0, 0, 0, 0, 0], 0
        
        # 恢复位姿
        _, R, t, pose_mask = cv2.recoverPose(E, pts_prev, pts_curr, self.K, mask=mask)
        
        self.num_inliers = np.sum(mask)

        # --- 改动2: RANSAC内点数阈值 — 外点过多则拒绝本次VO ---
        if self.num_inliers < 20:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return [0, 0, 0, 0, 0, 0], 0

        # 累积变换（简化版本，实际应该根据场景估计尺度）
        # 注意：单目VO无法恢复真实尺度，这里使用固定尺度
        self.t_total += self.scale * (self.R_total @ t)
        self.R_total = R @ self.R_total
        
        # 转换为欧拉角（ZYX顺序）
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        # 相对位姿（帧间运动）
        delta_x = float(self.scale * t[0, 0])
        delta_y = float(self.scale * t[1, 0])
        delta_z = float(self.scale * t[2, 0])
        
        # 更新上一帧
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des
        
        return [delta_x, delta_y, delta_z, roll, pitch, yaw], self.num_inliers
    
    def get_absolute_pose(self):
        """
        获取累积的绝对位姿
        
        返回:
            pose: [x, y, z, roll, pitch, yaw]
        """
        # 从旋转矩阵提取欧拉角
        R = self.R_total
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        if sy > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return [
            float(self.t_total[0, 0]),
            float(self.t_total[1, 0]),
            float(self.t_total[2, 0]),
            roll, pitch, yaw
        ]
    
    def reset(self):
        """重置视觉里程计"""
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        self.num_inliers = 0


class ScaleEstimator:
    """单目尺度估计器（使用IMU辅助）"""
    
    def __init__(self, alpha=0.95, use_fixed_scale=False, fixed_scale_value=1.0):
        """
        初始化尺度估计器
        
        参数:
            alpha: 平滑系数（更高=更保守）
            use_fixed_scale: 是否使用固定尺度（推荐True）
            fixed_scale_value: 固定尺度值
        """
        self.alpha = alpha
        self.scale_history = []
        self.current_scale = fixed_scale_value
        self.use_fixed_scale = use_fixed_scale
        self.fixed_scale_value = fixed_scale_value
    
    def estimate_scale(self, visual_translation, imu_translation):
        """
        估计尺度因子
        
        参数:
            visual_translation: 视觉估计的平移 [dx, dy, dz]
            imu_translation: IMU估计的平移 [dx, dy, dz]
        
        返回:
            scale: 尺度因子
        """
        # 如果使用固定尺度，直接返回
        if self.use_fixed_scale:
            return self.fixed_scale_value
        
        visual_norm = np.linalg.norm(visual_translation)
        imu_norm = np.linalg.norm(imu_translation)
        
        if visual_norm < 1e-6:
            return self.current_scale
        
        # 计算尺度
        scale = imu_norm / visual_norm
        
        # 更严格的尺度限制（防止崩溃）
        scale = np.clip(scale, 0.05, 5.0)  # 从[0.1,10]改为[0.5,2]
        
        # 更保守的平滑更新
        self.current_scale = self.alpha * self.current_scale + (1 - self.alpha) * scale
        self.scale_history.append(self.current_scale)
        
        # 保留最近100个尺度估计
        if len(self.scale_history) > 100:
            self.scale_history.pop(0)
        
        return self.current_scale
    
    def get_current_scale(self):
        """获取当前尺度"""
        return self.current_scale
