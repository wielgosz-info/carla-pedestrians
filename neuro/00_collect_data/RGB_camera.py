import os
import shutil
import sys
import carla
import random
import queue
import cv2
import numpy as np
import time
import math
import weakref
import collections

current_dir = os.path.dirname(os.path.abspath(__file__))
carla_api_path = os.path.join(current_dir, '../../../../carla-0.9.15/PythonAPI/carla')
sys.path.append(carla_api_path)
from agents.navigation.behavior_agent import BehaviorAgent

# -------------------------- 配置参数 --------------------------
TARGET_MAP = "Town10HD"
MAX_SAVE_IMG = 5000
OUTPUT_DIR = '../data/01_NeuroSLAM_Datasets/test/'
STAGNANT_SPEED = 0.1
STAGNANT_COUNT = 100
EXPOSURE_MODE = "manual"
EXPOSURE_COMPENSATION = "0.0"
FSTOP = "4.0"
ISO = "250"
GAMMA = "2.2"
QUEUE_MAXSIZE = 10
AGENT_BEHAVIOR = "normal"
AGENT_MAX_SPEED = 30  # km/h

# -------------------------- 全局变量 --------------------------
camera_transform = carla.Transform(
    carla.Location(x=0.2, y=0, z=4.2),
    carla.Rotation(pitch=-20)
)
rgb_image_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
imu_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
stagnant_count = 0

# -------------------------- 工具函数 --------------------------
def get_actor_display_name(actor, truncate=250):
    """获取Actor显示名称"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def safe_destroy_actor(actor):
    """对齐官方销毁逻辑"""
    if actor and actor.is_alive:
        if isinstance(actor, carla.Sensor):
            try:
                actor.stop()
                time.sleep(0.05)
            except Exception as e:
                print(f"停止传感器警告: {e}")
        try:
            actor.destroy()
        except Exception as e:
            print(f"销毁Actor警告: {e}")
    return None

# -------------------------- World类（核心优化） --------------------------
class World(object):
    """统一线程和资源管理"""
    def __init__(self, carla_world):
        self.world = carla_world
        self.map = self.world.get_map()
        self.player = None
        self.camera = None
        self.imu_sensor = None
        self.collision_sensor = None
        self.agent = None
        self.traffic_manager = None
        self.spectator = self.world.get_spectator()
        
        # 初始化环境设置
        self._init_world_settings()
        # 初始化输出目录
        self._init_output_dir()
        # 获取生成点
        self.spawn_points = self.map.get_spawn_points()
        print(f"找到 {len(self.spawn_points)} 个有效生成点")

    def _init_world_settings(self):
        """世界设置初始化"""
        self.traffic_manager = carla.Client('localhost', 2000).get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        print("已启用同步模式，步长: 0.05s")
        
        # 交通灯设置
        try:
            traffic_lights = self.world.get_actors().filter('traffic.traffic_light*')
            for tl in traffic_lights:
                tl.set_state(carla.TrafficLightState.Green)
                tl.freeze(True)
            print("所有交通灯已设置为绿灯并冻结")
        except Exception as e:
            print(f"配置交通灯警告: {e}")

    def _init_output_dir(self):
        """保留数据采集逻辑"""
        try:
            if os.path.exists(OUTPUT_DIR):
                shutil.rmtree(OUTPUT_DIR)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, 'IMU.txt'), 'w', encoding='utf-8') as f:
                f.write("timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
            print(f"输出目录准备完成: {OUTPUT_DIR}")
        except Exception as e:
            raise PermissionError(f"操作输出目录失败: {e}")

    def restart(self):
        """车辆重启（核心修复：添加生成后静置稳定）"""
        # 销毁原有资源
        self.destroy_sensors()
        if self.player:
            safe_destroy_actor(self.player)
        
        # 选择车辆蓝图
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        vehicle_bp.set_attribute('role_name', 'hero')
        
        # 生成车辆（多次尝试）
        self.player = None
        while self.player is None:
            spawn_point = random.choice(self.spawn_points)
            self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        self.modify_vehicle_physics(self.player)
        print(f"车辆生成成功，位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        
        # -------------------------- 核心修复：生成后静置稳定 --------------------------
        # 强制车辆初始姿态归零（刹车+零转向）
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0  # 刹车固定车身，避免物理抖动
        self.player.apply_control(control)
        # 通过tick推进物理引擎，等待姿态稳定（关键！）
        for _ in range(20):  # 等待20帧（1秒），确保物理引擎完全稳定
            self.world.tick()
        # 释放刹车，准备行驶
        control.brake = 0.0
        self.player.apply_control(control)
        self.world.tick()  # 再tick一次，应用控制
        # ------------------------------------------------------------------------------------------
        
        # Agent初始化（无自定义参数修改）
        self.agent = BehaviorAgent(self.player, behavior=AGENT_BEHAVIOR)
        self.agent.follow_speed_limits(True)
        #设置速度（CARLA 0.9.15兼容：通过局部规划器set_speed）
        self.agent._local_planner.set_speed(AGENT_MAX_SPEED / 3.6)
        
        # 设置目标点
        destination = random.choice(self.spawn_points).location
        self.agent.set_destination(destination)
        print(f"避障智能体初始化完成，目标: ({destination.x:.1f}, {destination.y:.1f})")
        
        # 传感器初始化
        self._init_sensors()
        
        # 观察者视角
        self.update_spectator()

    def modify_vehicle_physics(self, actor):
        """车辆物理配置"""
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception as e:
            print(f"配置车辆物理参数警告: {e}")

    def _init_sensors(self):
        """传感器初始化"""
        bp_lib = self.world.get_blueprint_library()
        
        # RGB相机
        self.camera = safe_destroy_actor(self.camera)
        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute("image_size_x", "640")
        rgb_bp.set_attribute("image_size_y", "480")
        rgb_bp.set_attribute("sensor_tick", "0.05")
        rgb_bp.set_attribute("exposure_mode", EXPOSURE_MODE)
        rgb_bp.set_attribute("exposure_compensation", EXPOSURE_COMPENSATION)
        rgb_bp.set_attribute("fstop", FSTOP)
        rgb_bp.set_attribute("iso", ISO)
        rgb_bp.set_attribute("gamma", GAMMA)
        rgb_bp.set_attribute("bloom_intensity", "0.0")
        rgb_bp.set_attribute("chromatic_aberration_intensity", "0.0")
        rgb_bp.set_attribute("lens_flare_intensity", "0.0")
        self.camera = self.world.spawn_actor(rgb_bp, camera_transform, attach_to=self.player)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda img: self._image_callback(img, weak_self))
        print("RGB相机初始化完成")
        
        # IMU传感器
        self.imu_sensor = safe_destroy_actor(self.imu_sensor)
        imu_bp = bp_lib.find("sensor.other.imu")
        imu_bp.set_attribute('sensor_tick', '0.1')
        imu_std_dev_a = 0.1
        imu_std_dev_g = 0.001
        imu_bp.set_attribute('noise_accel_stddev_x', str(imu_std_dev_a))
        imu_bp.set_attribute('noise_accel_stddev_y', str(imu_std_dev_a))
        imu_bp.set_attribute('noise_accel_stddev_z', str(imu_std_dev_a))
        imu_bp.set_attribute('noise_gyro_stddev_x', str(imu_std_dev_g))
        imu_bp.set_attribute('noise_gyro_stddev_y', str(imu_std_dev_g))
        imu_bp.set_attribute('noise_gyro_stddev_z', str(imu_std_dev_g))
        self.imu_sensor = self.world.spawn_actor(imu_bp, camera_transform, attach_to=self.player)
        self.imu_sensor.listen(lambda data: self._imu_callback(data, weak_self))
        print("IMU传感器初始化完成")
        
        # 碰撞传感器
        self.collision_sensor = CollisionSensor(self.player)
        print("碰撞传感器初始化完成")

    def _image_callback(self, image, weak_self):
        """通过弱引用访问全局队列"""
        self = weak_self()
        if not self or rgb_image_queue.full():
            return
        try:
            img_array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))[:, :, :3]
            rgb_image_queue.put((image.timestamp, img_array))
        except Exception as e:
            print(f"图像回调错误: {e}")

    def _imu_callback(self, data, weak_self):
        """线程安全"""
        self = weak_self()
        if not self or imu_queue.full():
            return
        try:
            imu_queue.put((data.timestamp, data))
        except Exception as e:
            print(f"IMU回调错误: {e}")

    def update_spectator(self):
        """保留观察者视角更新"""
        try:
            vehicle_tf = self.player.get_transform()
            spectator_tf = carla.Transform(
                vehicle_tf.transform(carla.Location(x=-4, z=50)),
                carla.Rotation(yaw=-180, pitch=-90)
            )
            self.spectator.set_transform(spectator_tf)
        except Exception as e:
            print(f"更新观察者位置失败: {e}")

    def destroy_sensors(self):
        """传感器销毁"""
        self.camera = safe_destroy_actor(self.camera)
        self.imu_sensor = safe_destroy_actor(self.imu_sensor)
        if self.collision_sensor:
            self.collision_sensor.sensor = safe_destroy_actor(self.collision_sensor.sensor)

    def destroy(self):
        """资源清理"""
        self.destroy_sensors()
        safe_destroy_actor(self.player)
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(False)

# -------------------------- 传感器类 --------------------------
class CollisionSensor(object):
    """CollisionSensor"""
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
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
        global stagnant_count
        actor_type = get_actor_display_name(event.other_actor)
        print(f"⚠️  碰撞检测：与 {actor_type} 发生碰撞")
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        stagnant_count += 30

# -------------------------- 数据保存函数 --------------------------
def save_image(output_dir, idx, img_array, target_width=160, target_height=120):
    try:
        resized_img = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        img_path = os.path.join(output_dir, f"{idx:04d}.png")
        cv2.imwrite(img_path, resized_img)
        return True
    except Exception as e:
        print(f"保存图像{idx}失败: {e}")
        return False

def save_imu(output_dir, timestamp, imu_data):
    try:
        imu_line = f"{timestamp:.6f},{imu_data.accelerometer.x:.6f},{imu_data.accelerometer.y:.6f},{imu_data.accelerometer.z:.6f}," \
                   f"{imu_data.gyroscope.x:.6f},{imu_data.gyroscope.y:.6f},{imu_data.gyroscope.z:.6f}\n"
        with open(os.path.join(output_dir, 'IMU.txt'), 'a+', encoding='utf-8') as f:
            f.write(imu_line)
        return True
    except Exception as e:
        print(f"保存IMU数据失败: {e}")
        return False

# -------------------------- 主循环 --------------------------
def main():
    global stagnant_count
    save_idx = 0
    img_idx = 0

    # 连接服务器+加载地图
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    try:
        carla_world = client.get_world()
        print(f"成功连接Carla服务器，当前地图: {carla_world.get_map().name}")
    except Exception as e:
        raise ConnectionError(f"连接Carla失败: {e}\n请先启动服务器：./CarlaUE4.sh -RenderOffScreen")
    
    print(f"加载指定地图 {TARGET_MAP}...")
    client.load_world(TARGET_MAP)
    time.sleep(3)
    carla_world = client.get_world()
    print(f"地图加载完成: {carla_world.get_map().name}")

    # 初始化World对象
    world = World(carla_world)
    world.restart()

    # 保留图像显示窗口
    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
    print("主循环启动，按 'q' 退出")

    try:
        while True:
            # 官方核心：同步模式tick
            world.world.tick()

            # 车辆状态检查（保留停滞检测）
            if world.player and world.player.is_alive:
                vel = world.player.get_velocity()
                speed = math.sqrt(vel.x**2 + vel.y**2)

                # 停滞检测逻辑保留
                if speed < STAGNANT_SPEED:
                    stagnant_count += 1
                    if stagnant_count >= STAGNANT_COUNT:
                        print(f"车辆停滞超过5秒（速度: {speed:.2f}m/s），重新生成...")
                        world.restart()
                        stagnant_count = 0
                else:
                    stagnant_count = 0

                # Agent控制
                if world.agent and world.agent.done():
                    destination = random.choice(world.spawn_points).location
                    world.agent.set_destination(destination)
                    print(f"已到达，新目标：({destination.x:.1f}, {destination.y:.1f})")
                if world.agent:
                    control = world.agent.run_step()
                    control.manual_gear_shift = False
                    world.player.apply_control(control)

                # 更新观察者视角
                world.update_spectator()
            else:
                print("车辆丢失，重新生成...")
                world.restart()
                time.sleep(1)
                continue

            # 数据采集逻辑保留
            if not rgb_image_queue.empty():
                try:
                    img_ts, cur_img = rgb_image_queue.get(timeout=0.1)
                    cv2.imshow('RGB Camera', cur_img)
                    img_idx += 1
                    if img_idx % 5 == 0:
                        save_idx += 1
                        save_image(OUTPUT_DIR, save_idx, cur_img)
                        print(f"已保存 {save_idx}/{MAX_SAVE_IMG} 张图像")
                        
                        if save_idx >= MAX_SAVE_IMG:
                            print("达到最大保存数量，退出程序")
                            return
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"处理图像失败: {e}")
            
            # IMU数据处理
            if not imu_queue.empty():
                try:
                    imu_ts, cur_imu = imu_queue.get(timeout=0.1)
                    save_imu(OUTPUT_DIR, imu_ts, cur_imu)
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"处理IMU数据失败: {e}")
            
            # 退出逻辑：按键检测
            if cv2.waitKey(1) == ord('q'):
                print("用户按下退出键，程序退出")
                return
            
            # 控制循环频率（补充微延迟避免CPU占用过高）
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("用户强制中断程序")
    except Exception as e:
        print(f"主循环错误: {e}")
    finally:
        # 资源清理（确保所有Actor销毁）
        if 'world' in locals():
            world.destroy()
        cv2.destroyAllWindows()
        print("所有资源已清理，程序退出")


if __name__ == "__main__":
    main()
