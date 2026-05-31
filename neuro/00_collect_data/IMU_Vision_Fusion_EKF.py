import os
import sys
import shutil
import math
import random
import numpy as np
import cv2
import queue
import time
import carla
import weakref
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))
carla_api_path = os.path.join(current_dir, '../../../../carla/PythonAPI/carla')
sys.path.append(carla_api_path)
from agents.navigation.behavior_agent import BehaviorAgent

# 导入视觉里程计
from visual_odometry_opencv import VisualOdometry, ScaleEstimator

# -------------------------- 配置参数 --------------------------
TARGET_MAP = "Town01"
MAX_SAVE_IMG = 5000
OUTPUT_DIR = os.path.join(current_dir, '..', 'data', 'Town01Data_IMU_Fusion')

# IMU-视觉融合参数优化
IMU_SAMPLE_RATE = 60  # Hz
CAMERA_SAMPLE_RATE = 20  # Hz (1/0.05)

EXPOSURE_MODE = "manual"
EXPOSURE_COMPENSATION = "0.0" 
FSTOP = "4.0"
ISO = "250"  
GAMMA = "2.2"
AGENT_BEHAVIOR = "cautious"
AGENT_MAX_SPEED = 20  # km/h - 降低速度以提高安全性
AGENT_SAFE_DISTANCE = 5.0  # 安全距离（米）
COLLISION_RESET_THRESHOLD = 3  # 碰撞次数阈值，超过后重置车辆

# -------------------------- 增强型碰撞传感器 --------------------------
class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.collision_count = 0
        self.collision_history = []
        self.last_collision_time = 0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        current_time = time.time()
        # 防止同一碰撞事件被多次记录（0.5秒内的视为同一次）
        if current_time - self.last_collision_time > 0.5:
            self.collision_count += 1
            self.last_collision_time = current_time
            actor_type = event.other_actor.type_id.split('.')[-1]
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_history.append({
                'time': current_time,
                'actor': actor_type,
                'intensity': intensity
            })
            print(f"[COLLISION] [{self.collision_count}x]: {actor_type} (intensity: {intensity:.2f})")
    
    def reset_collision_count(self):
        """重置碰撞计数"""
        self.collision_count = 0
        self.collision_history = []
    
    def has_major_collision(self):
        """检测是否发生严重碰撞"""
        return self.collision_count >= COLLISION_RESET_THRESHOLD

# -------------------------- 全局工具函数 --------------------------
def clear_all_actors(world):
    for actor in world.get_actors().filter('vehicle.*.*'):
        try:
            actor.destroy()
        except Exception as e:
            print(f"清理车辆警告: {e}")
    for actor in world.get_actors().filter('sensor.*.*'):
        try:
            actor.stop()
            actor.destroy()
        except Exception as e:
            print(f"清理传感器警告: {e}")
    for actor in world.get_actors().filter('walker.*.*'):
        try:
            actor.destroy()
        except Exception as e:
            print(f"清理行人警告: {e}")
    time.sleep(1)

def select_forward_destination(vehicle, spawn_points, min_distance=50.0):
    """选择车辆前方的目标点，优先直行路径"""
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_forward = vehicle_transform.get_forward_vector()
    
    # 筛选前方的spawn点
    forward_points = []
    for sp in spawn_points:
        to_spawn = sp.location - vehicle_location
        distance = to_spawn.length()
        
        # 计算是否在前方（点积>0表示在前方）
        if distance > min_distance:
            direction = to_spawn / distance
            dot_product = vehicle_forward.x * direction.x + vehicle_forward.y * direction.y
            
            if dot_product > 0.7:  # 夹角小于45度认为是前方
                forward_points.append((sp.location, distance, dot_product))
    
    if forward_points:
        # 按照"前方程度"排序，选择最前方的点
        forward_points.sort(key=lambda x: x[2], reverse=True)
        return forward_points[0][0]
    else:
        # 如果没有前方点，选择距离最远的点
        farthest = max(spawn_points, key=lambda sp: (sp.location - vehicle_location).length())
        return farthest.location

def safe_spawn_vehicle(world, bp_lib, max_attempts=10):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise ValueError(f"地图 {TARGET_MAP} 未找到生成点！")
    print(f"地图 {TARGET_MAP} 找到 {len(spawn_points)} 个生成点")

    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle_bp.set_attribute('role_name', 'hero')
    vehicle = None
    for attempt in range(max_attempts):
        chosen_spawn = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, chosen_spawn)
        if vehicle is not None:
            print(f"第{attempt+1}次尝试成功，生成车辆")
            return vehicle, spawn_points
        print(f"第{attempt+1}次生成失败，重试...")
        time.sleep(1)
    raise RuntimeError(f"连续{max_attempts}次生成失败！")

# -------------------------- 初始化CARLA环境 --------------------------
def init_carla_environment():
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    try:
        world = client.get_world()
        print("成功连接CARLA服务器")
    except Exception as e:
        raise ConnectionError(f"连接失败: {e}\n请先启动服务器：./CarlaUE4.sh")
    
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    
    # 优化Traffic Manager参数以提高安全性
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)  # 增加跟车距离
    traffic_manager.set_random_device_seed(42)  # 固定随机种子，行为可复现
    
    clear_all_actors(world)
    
    print(f"加载地图 {TARGET_MAP}...")
    client.load_world(TARGET_MAP)
    time.sleep(3)
    world = client.get_world()
    print(f"地图加载完成: {world.get_map().name}")
    
    for tl in world.get_actors().filter('traffic.traffic_light*'):
        try:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        except Exception as e:
            print(f"配置交通灯警告: {e}")
    
    bp_lib = world.get_blueprint_library()
    vehicle, spawn_points = safe_spawn_vehicle(world, bp_lib)
    try:
        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)
    except Exception as e:
        print(f"配置车辆物理参数警告: {e}")
    
    agent = BehaviorAgent(vehicle, behavior=AGENT_BEHAVIOR)
    agent.follow_speed_limits(False)  # 禁用限速，使用自定义速度
    
    # 兼容不同CARLA版本的速度设置
    try:
        agent.set_max_speed(AGENT_MAX_SPEED / 3.6)  # km/h转m/s
    except AttributeError:
        # 旧版本使用set_target_speed
        try:
            agent.set_target_speed(AGENT_MAX_SPEED / 3.6)
        except AttributeError:
            # 直接设置内部属性
            agent._max_speed = AGENT_MAX_SPEED / 3.6
            print(f"使用备用方式设置速度: {AGENT_MAX_SPEED} km/h")
    
    # 增强避障参数（兼容性处理）
    try:
        # 设置更大的安全距离和更保守的驾驶参数
        if hasattr(agent, '_vehicle_controller') and agent._vehicle_controller is not None:
            if hasattr(agent._vehicle_controller, '_args_lateral_dict'):
                agent._vehicle_controller._args_lateral_dict['K_P'] = 0.8
                agent._vehicle_controller._args_lateral_dict['K_I'] = 0.02
                agent._vehicle_controller._args_lateral_dict['K_D'] = 0.0
        if hasattr(agent, '_min_distance'):
            agent._min_distance = AGENT_SAFE_DISTANCE
        if hasattr(agent, '_max_brake'):
            agent._max_brake = 0.8
    except (AttributeError, KeyError, TypeError) as e:
        print(f"警告：无法设置高级避障参数 ({e})，使用默认配置")
    
    # 选择前方较远的目标点，避免斜着走
    destination = select_forward_destination(vehicle, spawn_points)
    agent.set_destination(destination)
    print(f"避障智能体初始化完成（最大速度: {AGENT_MAX_SPEED} km/h，安全距离: {AGENT_SAFE_DISTANCE}m）")
    print(f"目标位置: ({destination.x:.1f}, {destination.y:.1f}, {destination.z:.1f})")
    
    collision_sensor = CollisionSensor(vehicle)
    
    try:
        if os.path.exists(OUTPUT_DIR):
            backup_dir = OUTPUT_DIR + '_backup_' + time.strftime('%Y%m%d_%H%M%S')
            shutil.move(OUTPUT_DIR, backup_dir)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"无权限操作目录: {OUTPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    return world, bp_lib, vehicle, spawn_points, world.get_spectator(), agent, traffic_manager, collision_sensor

# -------------------------- 传感器配置 --------------------------
class SensorData:
    def __init__(self, data_type, timestamp, data):
        self.data_type = data_type
        self.timestamp = timestamp
        self.data = data

# -------------------------- 核心修改：同步参考代码的相机配置及图像处理 --------------------------
def create_rgb_camera(world, bp_lib, vehicle, data_queue):
    rgb_bp = bp_lib.find('sensor.camera.rgb')
    # 完全同步参考代码的相机参数
    rgb_bp.set_attribute("image_size_x", "640")
    rgb_bp.set_attribute("image_size_y", "480")
    rgb_bp.set_attribute("sensor_tick", "0.05")
    rgb_bp.set_attribute("exposure_mode", EXPOSURE_MODE)
    rgb_bp.set_attribute("exposure_compensation", EXPOSURE_COMPENSATION)
    rgb_bp.set_attribute("fstop", FSTOP)
    rgb_bp.set_attribute("iso", ISO)  
    rgb_bp.set_attribute("gamma", GAMMA)
    # 新增参考代码中的图像增强关闭参数，避免色彩失真
    rgb_bp.set_attribute("bloom_intensity", "0.0")
    rgb_bp.set_attribute("chromatic_aberration_intensity", "0.0")
    rgb_bp.set_attribute("lens_flare_intensity", "0.0")
    
    transform = carla.Transform(
        carla.Location(x=0.2, y=0, z=4.2),  
        carla.Rotation(pitch=-20)
    )
    
    # 同步参考代码的图像回调逻辑：直接保存原始RGB数据，避免格式转换失真
    def image_callback(image):
        # 参考代码逻辑：不做ColorConverter转换，直接保留原始数据
        data_queue.put(SensorData('image', image.timestamp, image))
    
    camera = world.spawn_actor(rgb_bp, transform, attach_to=vehicle)
    camera.listen(image_callback)
    print("RGB相机初始化完成（已同步参考代码色彩参数）")
    return camera, transform

def create_imu_sensor(world, bp_lib, vehicle, data_queue, transform):
    imu_bp = bp_lib.find("sensor.other.imu")
    imu_bp.set_attribute('sensor_tick', str(1/60))
    imu_bp.set_attribute('noise_accel_stddev_x', '0.1')
    imu_bp.set_attribute('noise_gyro_stddev_x', '0.001')
    imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
    imu.listen(lambda data: data_queue.put(SensorData('imu', data.timestamp, data)))
    print("IMU传感器初始化完成")
    return imu

# -------------------------- 时间戳对齐 --------------------------
class TimeAligner:
    def __init__(self, time_threshold=0.02):
        self.time_threshold = time_threshold
        self.imu_buffer = []
        self.image_buffer = []
        self.max_buffer = 100

    def add_data(self, data):
        if data.data_type == 'imu':
            self.imu_buffer.append(data)
            self.imu_buffer.sort(key=lambda x: x.timestamp)
            if len(self.imu_buffer) > self.max_buffer:
                self.imu_buffer.pop(0)
        elif data.data_type == 'image':
            self.image_buffer.append(data)
            self.image_buffer.sort(key=lambda x: x.timestamp)
            if len(self.image_buffer) > self.max_buffer//10:
                self.image_buffer.pop(0)

    def get_aligned_pairs(self):
        pairs = []
        if not self.image_buffer or not self.imu_buffer:
            return pairs
        for img in self.image_buffer:
            diffs = [abs(img.timestamp - imu.timestamp) for imu in self.imu_buffer]
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= self.time_threshold:
                pairs.append((img, self.imu_buffer[min_idx]))
                del self.imu_buffer[min_idx]
        self.image_buffer = []
        return pairs

# -------------------------- EKF融合--------------------------
class EKF_VIO:
    def __init__(self, init_pose, init_vel, dt=0.05):
        self.x = np.array([
            init_pose[0], init_pose[1], init_pose[2],
            init_vel[0], init_vel[1], init_vel[2],
            init_pose[3], init_pose[4], init_pose[5]
        ], dtype=np.float64)
        self.dt = dt
        self.init_z = init_pose[2]  # 保存初始Z轴高度（平地车辆固定）
        
        # EKF协方差矩阵 - 平衡IMU预测和VO观测
        self.P = np.diag([0.1, 0.1, 0.01, 0.5, 0.5, 0.1, 0.01, 0.01, 0.01])
        # 过程噪声（dt=0.05s, 合理取值避免P过度膨胀）
        self.Q = np.diag([0.05, 0.05, 0.001, 0.2, 0.2, 0.1, 0.005, 0.005, 0.005])
        # VO观测噪声 (R_base保存原始值, visual_update中自适应调整)
        self.R_base = np.diag([0.5, 0.5, 0.1, 0.05, 0.05, 0.05])
        self.R = self.R_base.copy()

        # 自适应R: EMA平滑的归一化新息平方
        self._nis_ema = 1.0

        # IMU零偏估计 (前庭静息校准)
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)
        self._bias_samples = 0
        self._bias_max_samples = 200  # 前200个静止样本用于初始校准
        
        # 统计信息
        self.innovation_history = []
        self.uncertainty_history = []
        self.mahalanobis_history = []

        # 新息门控：宽松阈值，仅过滤极端异常（P+=Q下P不会爆炸）
        self.chi2_threshold = 1000.0  # 宽松门控，仅拒绝极端异常
        self.innovation_accepted = 0
        self.innovation_rejected = 0

    def _gate_innovation(self, y, S):
        """Mahalanobis distance gate: returns (accepted, distance)."""
        try:
            S_inv = np.linalg.inv(S)
            d = float(y.T @ S_inv @ y)
            return d < self.chi2_threshold, d
        except np.linalg.LinAlgError:
            return False, float('inf')

    def _adapt_R(self, y, S):
        """自适应R: 基于归一化新息平方(NIS)动态调整VO观测噪声。
        NIS高=VO残差异常=R增大(降低VO权重), NIS低=VO可靠=R减小(信任VO修正IMU)"""
        try:
            S_inv = np.linalg.inv(S + np.eye(6)*1e-8)
            nis = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            return
        expected_nis = 6.0
        self._nis_ema = 0.9 * self._nis_ema + 0.1 * (nis / expected_nis)
        scale = np.clip(self._nis_ema, 0.2, 10.0)
        self.R = self.R_base * scale

    def _estimate_imu_bias(self, accel, gyro):
        """IMU零偏估计: 当加速度接近重力(车辆静止/匀速)时, 累积样本估计零偏。
        生物启发: 前庭系统静息校准 — 静止时学习补偿基线偏移"""
        accel_mag = np.linalg.norm(accel)
        gyro_mag = np.linalg.norm(gyro)
        # 静止判断: 加速度≈9.81且角速度≈0 (重力是唯一加速度)
        is_static = (abs(accel_mag - 9.81) < 0.5) and (gyro_mag < 0.02)
        if is_static and self._bias_samples < self._bias_max_samples:
            alpha = 1.0 / (self._bias_samples + 1)
            self._gyro_bias = (1 - alpha) * self._gyro_bias + alpha * gyro
            self._accel_bias = (1 - alpha) * self._accel_bias + alpha * (accel - self._gravity_dir(accel_mag))
            self._bias_samples += 1

    @staticmethod
    def _gravity_dir(mag):
        """估计重力方向: 假设加速度计测量值的主方向即重力方向"""
        return np.array([0.0, 0.0, mag])

    def _numerical_jacobian(self, imu_data, epsilon=1e-6):
        """State transition Jacobian (9x9) via central finite differences.
        Handles angle wrapping for orientation states 6-8."""
        accel = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z])
        gyro = np.array([imu_data.gyroscope.x,
                         imu_data.gyroscope.y,
                         imu_data.gyroscope.z])
        dt = self.dt
        decay = 0.98

        def f_state(x):
            roll, pitch, yaw = x[6], x[7], x[8]
            new_roll  = (roll  + gyro[0]*dt + np.pi) % (2*np.pi) - np.pi
            new_pitch = (pitch + gyro[1]*dt + np.pi) % (2*np.pi) - np.pi
            new_yaw   = (yaw   + gyro[2]*dt + np.pi) % (2*np.pi) - np.pi
            R_mat = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
            a_w = R_mat @ accel
            new_vx = decay * (x[3] + a_w[0] * dt)
            new_vy = decay * (x[4] + a_w[1] * dt)
            new_px = x[0] + new_vx * dt
            new_py = x[1] + new_vy * dt
            return np.array([new_px, new_py, x[2],
                             new_vx, new_vy, 0.0,
                             new_roll, new_pitch, new_yaw])

        x0 = self.x.copy()
        F = np.zeros((9, 9))
        for i in range(9):
            xp = x0.copy(); xp[i] += epsilon
            xm = x0.copy(); xm[i] -= epsilon
            fp = f_state(xp)
            fm = f_state(xm)
            diff = fp - fm
            # Unwrap angular differences for orientation states
            for j in range(6, 9):
                if diff[j] > np.pi:
                    diff[j] -= 2*np.pi
                elif diff[j] < -np.pi:
                    diff[j] += 2*np.pi
            F[:, i] = diff / (2.0 * epsilon)
        return F

    def imu_prediction(self, imu_data):
        accel_raw = np.array([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z])
        gyro_raw = np.array([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z])

        # --- 改动3: IMU零偏补偿 (前庭静息校准) ---
        self._estimate_imu_bias(accel_raw, gyro_raw)
        gyro = gyro_raw - self._gyro_bias
        accel = accel_raw - self._accel_bias
        # --- 零偏补偿结束 ---

        roll, pitch, yaw = self.x[6], self.x[7], self.x[8]
        new_roll = (roll + gyro[0]*self.dt + np.pi) % (2*np.pi) - np.pi
        new_pitch = (pitch + gyro[1]*self.dt + np.pi) % (2*np.pi) - np.pi
        new_yaw = (yaw + gyro[2]*self.dt + np.pi) % (2*np.pi) - np.pi

        R_body2world = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        accel_world = R_body2world @ accel
        
        # 添加速度衰减因子，防止IMU积分漂移
        velocity_decay = 0.98  # 每帧衰减2%
        new_vx = velocity_decay * (self.x[3] + accel_world[0] * self.dt)
        new_vy = velocity_decay * (self.x[4] + accel_world[1] * self.dt)
        # Z轴：平地行驶不使用加速度计（包含重力，会导致漂移）
        new_vz = 0.0  # 地面车辆Z轴速度为0

        new_x = self.x[0] + new_vx * self.dt
        new_y = self.x[1] + new_vy * self.dt
        new_z = self.x[2]  # Z轴位置保持不变（平地行驶）

        self.x = np.array([new_x, new_y, new_z, new_vx, new_vy, new_vz, new_roll, new_pitch, new_yaw])
        self.P += self.Q  # 简化协方差传播（避免数值雅可比在大坐标下发散）

    def visual_update(self, visual_pose):
        z = np.array(visual_pose, dtype=np.float64)
        H = np.zeros((6, 9))
        H[0,0] = H[1,1] = H[2,2] = 1
        H[3,6] = H[4,7] = H[5,8] = 1

        y = z - H @ self.x  # 新息(innovation)
        S = H @ self.P @ H.T + self.R + np.eye(6)*1e-8

        # 新息门控：卡方检验拒绝异常VO观测
        accepted, mahal_dist = self._gate_innovation(y, S)
        self.mahalanobis_history.append(mahal_dist)
        if not accepted:
            self.innovation_rejected += 1
            self.innovation_history.append(np.linalg.norm(y))
            return  # 拒绝本次VO更新

        self.innovation_accepted += 1
        self.innovation_history.append(np.linalg.norm(y))
        self.uncertainty_history.append(np.trace(self.P[:3, :3]))

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.eye(9, 6) * 0.1

        self.x += K @ y
        # Joseph形式协方差更新，保证数值稳定性
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5*(self.P + self.P.T)  # 保持对称性

        # --- 改动1: 自适应R — 根据本次残差调整下次观测噪声 ---
        self._adapt_R(y, S)

        # Soft flat-ground pseudo-measurement (replaces hard clamp)
        z_flat = np.array([self.init_z, 0.0, 0.0])  # pz=init_z, roll=0, pitch=0
        H_flat = np.zeros((3, 9))
        H_flat[0, 2] = 1.0   # selects pz
        H_flat[1, 6] = 1.0   # selects roll
        H_flat[2, 7] = 1.0   # selects pitch
        R_flat = np.diag([0.01, 0.001, 0.001])

        y_flat = z_flat - H_flat @ self.x
        S_flat = H_flat @ self.P @ H_flat.T + R_flat + np.eye(3) * 1e-8
        try:
            K_flat = self.P @ H_flat.T @ np.linalg.inv(S_flat)
        except np.linalg.LinAlgError:
            K_flat = np.zeros((9, 3))
        self.x += K_flat @ y_flat
        I_KH_flat = np.eye(9) - K_flat @ H_flat
        self.P = I_KH_flat @ self.P @ I_KH_flat.T + K_flat @ R_flat @ K_flat.T
        self.P = 0.5 * (self.P + self.P.T)

    def get_current_pose(self):
        return self.x[:3].copy(), self.x[6:9].copy()
    
    def get_current_velocity(self):
        return self.x[3:6].copy()
    
    def get_position_uncertainty(self):
        """返回位置估计的不确定性(标准差)"""
        return np.sqrt(np.diag(self.P[:3, :3]))
    
    def get_fusion_quality_metrics(self):
        """返回融合质量指标"""
        total = self.innovation_accepted + self.innovation_rejected
        rejection_rate = self.innovation_rejected / total if total > 0 else 0.0
        n_innov = min(len(self.innovation_history), 100)
        n_uncert = min(len(self.uncertainty_history), 100)
        n_mahal = min(len(self.mahalanobis_history), 100)
        return {
            'avg_innovation': float(np.mean(self.innovation_history[-n_innov:])) if n_innov > 0 else 0.0,
            'avg_uncertainty': float(np.mean(self.uncertainty_history[-n_uncert:])) if n_uncert > 0 else 0.0,
            'innovation_std': float(np.std(self.innovation_history[-n_innov:])) if n_innov > 1 else 0.0,
            'rejection_rate': rejection_rate,
            'avg_mahalanobis': float(np.mean(self.mahalanobis_history[-n_mahal:])) if n_mahal > 0 else 0.0,
        }

# -------------------------- 图像后处理 --------------------------
def save_image_simple(img_array, output_dir, idx, target_width=160, target_height=120):
    """缩放为160×120，同步参考代码色彩处理逻辑"""
    try:
        # 参考代码逻辑：直接使用RGB数据，不做RGBA2BGR转换
        resized_img = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        img_path = os.path.join(output_dir, f"{idx:04d}.png")
        cv2.imwrite(img_path, resized_img)
        return True
    except Exception as e:
        print(f"保存图像{idx}失败: {e}")
        return False

# -------------------------- 主循环 --------------------------
def main():
    try:
        world, bp_lib, vehicle, spawn_points, spectator, agent, traffic_manager, collision_sensor = init_carla_environment()
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    sensor_queue = queue.Queue(maxsize=1000)
    try:
        camera, cam_transform = create_rgb_camera(world, bp_lib, vehicle, sensor_queue)
        imu = create_imu_sensor(world, bp_lib, vehicle, sensor_queue, cam_transform)
    except Exception as e:
        print(f"传感器初始化失败: {e}")
        clear_all_actors(world)
        return

    time_aligner = TimeAligner()
    init_pose = vehicle.get_transform()
    init_pos = [init_pose.location.x, init_pose.location.y, init_pose.location.z]
    init_att = [math.radians(init_pose.rotation.roll),
                math.radians(init_pose.rotation.pitch),
                math.radians(init_pose.rotation.yaw)]
    ekf = EKF_VIO(init_pos+init_att, [0,0,0])
    
    # ============ 初始化真正的视觉里程计 ============
    visual_odom = VisualOdometry()
    # 使用固定尺度=1.0（VO本身尺度已经接近正确）
    scale_estimator = ScaleEstimator(alpha=0.95, use_fixed_scale=False, fixed_scale_value=1.0)
    
    # 用于累积视觉位姿（相对初始位置）
    vo_x, vo_y, vo_z = 0.0, 0.0, 0.0
    vo_roll, vo_pitch, vo_yaw = 0.0, 0.0, 0.0
    
    # 保存视觉里程计轨迹
    vo_log = open(os.path.join(OUTPUT_DIR, 'visual_odometry.txt'), 'w', encoding='utf-8')
    vo_log.write("timestamp,vo_x,vo_y,vo_z,vo_roll,vo_pitch,vo_yaw,num_matches,scale\n")

    # 数据保存参数（每1帧对齐数据保存1帧，确保时间戳同步）
    img_idx = 0
    fusion_log = open(os.path.join(OUTPUT_DIR, 'fusion_pose.txt'), 'w', encoding='utf-8')
    fusion_log.write("timestamp,pos_x,pos_y,pos_z,roll,pitch,yaw,imu_pos_x,imu_pos_y,imu_pos_z,vel_x,vel_y,vel_z,uncertainty_x,uncertainty_y,uncertainty_z\n")
    
    # 保存Ground Truth（CARLA车辆真实位置）
    gt_log = open(os.path.join(OUTPUT_DIR, 'ground_truth.txt'), 'w', encoding='utf-8')
    gt_log.write("timestamp,pos_x,pos_y,pos_z,roll,pitch,yaw,vel_x,vel_y,vel_z\n")
    
    # 添加MATLAB兼容的元数据文件
    metadata_log = open(os.path.join(OUTPUT_DIR, 'dataset_metadata.txt'), 'w', encoding='utf-8')
    metadata_log.write(f"# IMU-Visual SLAM Dataset\n")
    metadata_log.write(f"# Map: {TARGET_MAP}\n")
    metadata_log.write(f"# Camera Rate: {CAMERA_SAMPLE_RATE} Hz\n")
    metadata_log.write(f"# IMU Rate: {IMU_SAMPLE_RATE} Hz\n")
    metadata_log.write(f"# Image Size: 160x120\n")
    metadata_log.write(f"# Total Frames: {MAX_SAVE_IMG}\n")
    metadata_log.close()

    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
    stagnant_count = 0
    first_valid_imu = False  # 标记是否已跳过初始异常IMU帧

    # 提前打开对齐IMU文件句柄，避免循环内重复open
    aligned_imu_path = os.path.join(OUTPUT_DIR, 'aligned_imu.txt')
    aligned_imu_f = open(aligned_imu_path, 'w', encoding='utf-8')
    aligned_imu_f.write("timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")

    try:
        while True:                                                                                                  
            world.tick()
    
            while not sensor_queue.empty():
                try:
                    data = sensor_queue.get(block=False)
                    time_aligner.add_data(data)
                except:
                    break


            # 处理对齐的图像和IMU数据
            for img_data, imu_data in time_aligner.get_aligned_pairs():
                # 检查IMU数据有效性（跳过CARLA传感器异常帧，每帧都检查）
                accel_mag = math.sqrt(
                    imu_data.data.accelerometer.x**2 +
                    imu_data.data.accelerometer.y**2 +
                    imu_data.data.accelerometer.z**2
                )
                # 正常重力约9.8 m/s²，异常帧可达几万 m/s²；同时过滤NaN
                if accel_mag > 100.0 or math.isnan(accel_mag):
                    if first_valid_imu:
                        print(f"[WARN] Skipping abnormal IMU frame: accel_mag={accel_mag:.1f} m/s^2 ({accel_mag/9.8:.0f}G)")
                    continue
                first_valid_imu = True
                
                ekf.imu_prediction(imu_data.data)
                imu_pos, imu_att = ekf.get_current_pose()

                # 纯IMU速度积分（地面车辆只用XY，Z分量含重力不参与尺度估计）
                accel_body = np.array([imu_data.data.accelerometer.x,
                                       imu_data.data.accelerometer.y,
                                       imu_data.data.accelerometer.z])
                _, raw_att = ekf.get_current_pose()
                R_raw = R.from_euler('xyz', raw_att).as_matrix()
                accel_world = R_raw @ accel_body
                accel_world[2] = 0.0  # 去掉重力分量（地面车辆Z加速度为零）
                if not hasattr(main, '_raw_imu_vel'):
                    main._raw_imu_vel = np.zeros(3)
                main._raw_imu_vel = 0.98 * (main._raw_imu_vel + accel_world * ekf.dt)
                main._raw_imu_vel[2] = 0.0  # Z轴速度强制为0
                raw_imu_delta = main._raw_imu_vel * ekf.dt

                # ============ 使用真正的视觉里程计，而非Ground Truth ============
                # 参考代码逻辑：raw_data为RGBA格式，取前3通道作为RGB
                img = np.reshape(np.copy(img_data.data.raw_data), (480, 640, 4))[:, :, :3]
                
                # 运行视觉里程计（返回相对运动）
                vo_result, num_matches = visual_odom.process_frame(img)
                
                if vo_result is not None:
                    # VO返回相机坐标系下的帧间delta: X=右, Y=下, Z=前
                    delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw = vo_result
                    cam_right = float(delta_x)    # 相机X -> 横向
                    cam_forward = float(delta_z)  # 相机Z -> 前向

                    # 使用IMU估计的位移来校准VO的尺度（解决单目尺度模糊）
                    if img_idx > 1:
                        prev_scale = visual_odom.scale
                        imu_delta = raw_imu_delta
                        vo_delta = np.array([cam_forward, cam_right, 0.0])
                        scale = scale_estimator.estimate_scale(vo_delta, imu_delta)
                        if abs(prev_scale) < 1e-9:
                            prev_scale = 1.0
                        cam_forward = (cam_forward / prev_scale) * scale
                        cam_right = (cam_right / prev_scale) * scale
                        visual_odom.scale = scale
                    else:
                        scale = 1.0
                        prev_scale = visual_odom.scale
                        if abs(prev_scale) < 1e-9:
                            prev_scale = 1.0
                        cam_forward = (cam_forward / prev_scale) * scale
                        cam_right = (cam_right / prev_scale) * scale

                    # 累积视觉位姿：从相机坐标系旋转到世界坐标系
                    vo_yaw += delta_yaw
                    vo_x += cam_forward * np.cos(vo_yaw) - cam_right * np.sin(vo_yaw)
                    vo_y += cam_forward * np.sin(vo_yaw) + cam_right * np.cos(vo_yaw)
                    vo_z = 0.0

                    # 构建视觉观测位姿（绝对位姿）
                    visual_pose = [
                        init_pos[0] + vo_x,
                        init_pos[1] + vo_y,
                        init_pos[2] + vo_z,
                        init_att[0],
                        init_att[1],
                        init_att[2] + vo_yaw
                    ]

                    # 保存视觉里程计数据（VO相对增量，不包含init偏移）
                    vo_log.write(f"{img_data.timestamp:.6f},"
                                f"{vo_x:.6f},{vo_y:.6f},{vo_z:.6f},"
                                f"0.0,0.0,{vo_yaw:.6f},"
                                f"{num_matches},{scale:.6f}\n")
                else:
                    # 首帧或特征不足，使用上一次的位姿
                    visual_pose = [
                        init_pos[0] + vo_x,
                        init_pos[1] + vo_y,
                        init_pos[2] + vo_z,
                        init_att[0],
                        init_att[1],
                        init_att[2] + vo_yaw
                    ]
                    num_matches = 0
                    scale = 1.0

                # EKF视觉更新（融合IMU预测和VO观测）
                ekf.visual_update(visual_pose)
                
                # 获取EKF融合后的位姿（真正的IMU-视觉融合）
                fusion_pos, fusion_att = ekf.get_current_pose()
                
                # ============ 保存Ground Truth（仅用于评估，不参与融合） ============
                carla_pose = vehicle.get_transform()
                fusion_vel = ekf.get_current_velocity()
                pos_uncertainty = ekf.get_position_uncertainty()
                
                # 保存Ground Truth（CARLA车辆真实位置）
                vehicle_vel = vehicle.get_velocity()
                gt_log.write(f"{img_data.timestamp:.6f},"
                            f"{carla_pose.location.x:.6f},{carla_pose.location.y:.6f},{carla_pose.location.z:.6f},"
                            f"{carla_pose.rotation.roll:.6f},{carla_pose.rotation.pitch:.6f},{carla_pose.rotation.yaw:.6f},"
                            f"{vehicle_vel.x:.6f},{vehicle_vel.y:.6f},{vehicle_vel.z:.6f}\n")
                
                # 每10帧flush一次ground truth数据和VO数据
                if img_idx % 10 == 0:
                    gt_log.flush()
                    vo_log.flush()

                # 保存图像（保持每1帧对齐数据保存1帧，确保与IMU时间戳同步）
                # 注意：img已经在VO处理时读取了
                img_idx += 1
                save_image_simple(img, OUTPUT_DIR, img_idx)  # 使用同步后的保存函数
                print(f"保存图像 {img_idx}/{MAX_SAVE_IMG}")

                # 保存IMU数据（与图像时间戳严格对齐）
                aligned_imu_f.write(f"{imu_data.timestamp:.6f},"
                        f"{imu_data.data.accelerometer.x:.6f},{imu_data.data.accelerometer.y:.6f},{imu_data.data.accelerometer.z:.6f},"
                        f"{imu_data.data.gyroscope.x:.6f},{imu_data.data.gyroscope.y:.6f},{imu_data.data.gyroscope.z:.6f}\n")

                # 保存融合结果（增加速度和不确定性信息）
                fusion_log.write(f"{img_data.timestamp:.6f},"
                                f"{fusion_pos[0]:.6f},{fusion_pos[1]:.6f},{fusion_pos[2]:.6f},"
                                f"{math.degrees(fusion_att[0]):.6f},{math.degrees(fusion_att[1]):.6f},{math.degrees(fusion_att[2]):.6f},"
                                f"{imu_pos[0]:.6f},{imu_pos[1]:.6f},{imu_pos[2]:.6f},"
                                f"{fusion_vel[0]:.6f},{fusion_vel[1]:.6f},{fusion_vel[2]:.6f},"
                                f"{pos_uncertainty[0]:.6f},{pos_uncertainty[1]:.6f},{pos_uncertainty[2]:.6f}\n")
                
                # 每10帧flush一次，确保数据及时写入磁盘
                if img_idx % 10 == 0:
                    fusion_log.flush()
                
                # 每100帧打印融合质量指标
                if img_idx % 100 == 0 and img_idx > 0:
                    metrics = ekf.get_fusion_quality_metrics()
                    print(f"\n[Fusion Quality] Frame {img_idx}:")
                    print(f"  EKF新息: {metrics['avg_innovation']:.4f}, 不确定性: {metrics['avg_uncertainty']:.4f}")
                    print(f"  VO特征: {num_matches}匹配点, 尺度: {scale:.4f}")
                    print(f"  已保存 {img_idx} 条融合位姿数据\n")

                if img_idx >= MAX_SAVE_IMG:
                    print("达到最大保存数量，退出")
                    return

                # 显示原始图像
                cv2.imshow('RGB Camera', img)

            # 增强型避障智能体控制
            if agent.done():
                destination = select_forward_destination(vehicle, spawn_points)
                agent.set_destination(destination)
                print(f"[OK] Destination reached, new target: ({destination.x:.1f}, {destination.y:.1f}, {destination.z:.1f})")
            
            try:
                control = agent.run_step()
                control.manual_gear_shift = False
                vehicle.apply_control(control)
            except Exception as e:
                print(f"警告：控制命令执行失败 - {e}")
                control = carla.VehicleControl()
                control.brake = 1.0
                vehicle.apply_control(control)

            # 碰撞检测与重置（优先级高）
            reset_needed = False
            reset_reason = ""
            
            if collision_sensor.has_major_collision():
                print(f"[WARN] Multiple collisions ({collision_sensor.collision_count}), resetting vehicle...")
                reset_needed = True
                reset_reason = "碰撞过多"
            
            # 停滞检测与重置（次要）
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            if speed < 0.1:
                stagnant_count += 1
                if stagnant_count > 150:  # 增加容忍度到150帧（约7.5秒）
                    print(f"[STUCK] Vehicle stagnant ({stagnant_count} frames), resetting...")
                    reset_needed = True
                    reset_reason = "停滞"
            else:
                stagnant_count = 0

            if reset_needed:
                print(f"[RESET] Starting reset (reason: {reset_reason})...")
                try:
                    # 清理旧传感器
                    camera.stop()
                    imu.stop()
                    collision_sensor.sensor.stop()
                    camera.destroy()
                    imu.destroy()
                    collision_sensor.sensor.destroy()
                    vehicle.destroy()
                    time.sleep(0.5)  # 等待清理完成
                    
                    # 生成新车辆
                    vehicle, _ = safe_spawn_vehicle(world, bp_lib)
                    
                    # 配置车辆物理参数
                    physics_control = vehicle.get_physics_control()
                    physics_control.use_sweep_wheel_collision = True
                    vehicle.apply_physics_control(physics_control)
                    
                    # 重新初始化智能体（使用更安全的配置）
                    agent = BehaviorAgent(vehicle, behavior=AGENT_BEHAVIOR)
                    agent.follow_speed_limits(False)  # 禁用限速
                    
                    # 兼容不同CARLA版本的速度设置
                    try:
                        agent.set_max_speed(AGENT_MAX_SPEED / 3.6)
                    except AttributeError:
                        try:
                            agent.set_target_speed(AGENT_MAX_SPEED / 3.6)
                        except AttributeError:
                            agent._max_speed = AGENT_MAX_SPEED / 3.6
                    
                    # 设置避障参数（兼容性处理）
                    try:
                        if hasattr(agent, '_vehicle_controller') and agent._vehicle_controller is not None:
                            if hasattr(agent._vehicle_controller, '_args_lateral_dict'):
                                agent._vehicle_controller._args_lateral_dict['K_P'] = 0.8
                                agent._vehicle_controller._args_lateral_dict['K_I'] = 0.02
                                agent._vehicle_controller._args_lateral_dict['K_D'] = 0.0
                        if hasattr(agent, '_min_distance'):
                            agent._min_distance = AGENT_SAFE_DISTANCE
                        if hasattr(agent, '_max_brake'):
                            agent._max_brake = 0.8
                    except (AttributeError, KeyError, TypeError):
                        pass
                    
                    # 选择前方目标点
                    destination = select_forward_destination(vehicle, spawn_points)
                    agent.set_destination(destination)
                    print(f"新目标: ({destination.x:.1f}, {destination.y:.1f}, {destination.z:.1f})")
                    
                    # 重新创建传感器
                    camera, cam_transform = create_rgb_camera(world, bp_lib, vehicle, sensor_queue)
                    imu = create_imu_sensor(world, bp_lib, vehicle, sensor_queue, cam_transform)
                    collision_sensor = CollisionSensor(vehicle)
                    
                    # 重置计数器
                    stagnant_count = 0
                    
                    print(f"[OK] Reset complete, continuing...")
                    
                except Exception as e:
                    print(f"[ERROR] Reset failed: {e}")
                    print("尝试继续运行...")

            # 跟随视角
            spec_transform = carla.Transform(
                vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            spectator.set_transform(spec_transform)

            if cv2.waitKey(1) == ord('q'):
                print("用户退出")
                return

    except Exception as e:
        print(f"主循环错误: {e}")
    finally:
        fusion_log.close()
        gt_log.close()
        vo_log.close()  # 关闭视觉里程计日志
        aligned_imu_f.close()
        camera.stop()
        imu.stop()
        collision_sensor.sensor.stop()
        clear_all_actors(world)
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(False)
        cv2.destroyAllWindows()
        print("资源清理完成")


if __name__ == "__main__":
    main()
