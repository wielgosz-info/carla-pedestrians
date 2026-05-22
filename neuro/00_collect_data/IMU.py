import matplotlib.pyplot as plt
import numpy as np
import time
import csv

import agent
import integrate
import kalman_filter

# -------------------------- 参数配置 --------------------------
imu_rate  = 60  # IMU采样率（Hz）
imu_per   = 1 / imu_rate
gnss_rate = 5   # GNSS采样率（Hz）
gnss_per  = 1 / gnss_rate
save_time = 100   # 采集时长（秒）
imu_len   = save_time * imu_rate
gnss_len  = save_time * gnss_rate

# 传感器噪声参数
imu_std_dev_a = 0.1    
imu_std_dev_g = 0.001  
gnss_std_dev_geo = 3e-5
gnss_std_dev_xy = (np.pi * gnss_std_dev_geo / 180) * 6.357e6
imu_var_a = 0.05
imu_var_g = 0.01
gnss_var = 30

print(f"GNSS std deviation : {gnss_std_dev_xy:.3f}m")

# -------------------------- 仿真环境与数据采集 --------------------------
a = agent.agent()
d = a.world.debug
a.spawn_vehicle(1)
a.vehicle.set_autopilot(True)

imu_list = []
gnss_list = []
real_pos = []

# IMU数据监听（新增终端打印）
def imu_listener(data):
    if len(imu_list) < imu_len:
        accel = data.accelerometer
        gyro = data.gyroscope
        imu_list.append(((accel.x, accel.y, accel.z), (gyro.x, gyro.y, gyro.z), data.timestamp))
        # 终端打印IMU实时数据
        print(f"IMU采集: 加速度({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}) 角速度({gyro.x:.3f}, {gyro.y:.3f}, {gyro.z:.3f}) 时间戳{data.timestamp:.2f}")

# GNSS数据监听（新增终端打印）
def gnss_listener(data):
    if len(gnss_list) < gnss_len:
        x, y, z = a.gnss_to_xyz(data)
        rpos = data.transform.location
        gnss_list.append(((x, y, z), data.timestamp))
        real_pos.append(((rpos.x, rpos.y, rpos.z), data.timestamp))
        # 终端打印GNSS实时数据
        print(f"GNSS采集: 位置({x:.2f}, {y:.2f}, {z:.2f}) 真实位置({rpos.x:.2f}, {rpos.y:.2f}, {rpos.z:.2f}) 时间戳{data.timestamp:.2f}")

a.spawn_imu(imu_per, imu_std_dev_a, imu_std_dev_g)
a.imu_reg_callback(imu_listener)
a.spawn_gnss(gnss_per, gnss_std_dev_geo)
a.gnss_reg_callback(gnss_listener)

# -------------------------- 纯IMU积分 --------------------------
init_vel = a.vehicle.get_velocity()
init_loc = a.vehicle.get_location()
init_rot = a.vehicle.get_transform().rotation
timestamp = a.world.get_snapshot().timestamp.elapsed_seconds
init_state = np.asarray([init_loc.x, init_loc.y, init_rot.yaw * np.pi / 180, init_vel.x, init_vel.y]).reshape(5, 1)
int_obj = integrate.imu_integrate(init_state, timestamp)

int_rvel_list = []
int_rpos_list = []
int_ryaw_list = []

def imu_int_listener(data):
    if len(int_rvel_list) < imu_len:
        rvel = a.vehicle.get_velocity()
        rloc = a.vehicle.get_location()
        rrot = a.vehicle.get_transform().rotation
        current_ts = data.timestamp
        int_rvel_list.append(((rvel.x, rvel.y), current_ts))
        int_rpos_list.append(((rloc.x, rloc.y), current_ts))
        int_ryaw_list.append((rrot.yaw * np.pi / 180, current_ts))
        yaw_vel = data.gyroscope.z
        accel_x = data.accelerometer.x
        accel_y = data.accelerometer.y
        int_obj.update(np.asarray([accel_x, accel_y, yaw_vel]).reshape(3, 1), current_ts)
        # 终端打印IMU积分实时状态
        print(f"IMU积分: 速度({rvel.x:.2f}, {rvel.y:.2f}) 位置({rloc.x:.2f}, {rloc.y:.2f}) 偏航角{rrot.yaw:.2f}° 时间戳{current_ts:.2f}")

a.imu_reg_callback(imu_int_listener)

# -------------------------- 卡尔曼滤波 --------------------------
kal_obj = kalman_filter.kalman_filter(init_state, timestamp, imu_var_a, imu_var_g, gnss_var)

kal_rvel_list = []
kal_rpos_list = []
kal_ryaw_list = []
kal_gnss_list = []
kal_gact_list = []

def imu_kal_listener(data):
    if len(kal_rvel_list) < imu_len:
        rvel = a.vehicle.get_velocity()
        rloc = a.vehicle.get_location()
        rrot = a.vehicle.get_transform().rotation
        current_ts = data.timestamp
        kal_rvel_list.append(((rvel.x, rvel.y), current_ts))
        kal_rpos_list.append(((rloc.x, rloc.y), current_ts))
        kal_ryaw_list.append((rrot.yaw * np.pi / 180, current_ts))
        yaw_vel = data.gyroscope.z
        accel_x = data.accelerometer.x
        accel_y = data.accelerometer.y
        kal_obj.update(np.asarray([accel_x, accel_y, yaw_vel]).reshape(3, 1), current_ts)
        # 终端打印卡尔曼滤波实时预测
        print(f"卡尔曼预测: 速度({rvel.x:.2f}, {rvel.y:.2f}) 位置({kal_obj.state[0,0]:.2f}, {kal_obj.state[1,0]:.2f}) 偏航角{kal_obj.state[2,0]*180/np.pi:.2f}° 时间戳{current_ts:.2f}")

def gnss_kal_listener(data):
    if len(kal_gnss_list) < gnss_len and len(kal_rvel_list) > 0:
        rloc = a.vehicle.get_location()
        gnss_ts = data.timestamp
        x, y, z = a.gnss_to_xyz(data)
        kal_gnss_list.append(((x, y), gnss_ts))
        kal_gact_list.append(((rloc.x, rloc.y), gnss_ts))
        prev_pos = agent.carla.Location(kal_obj.state[0][0], kal_obj.state[1][0], rloc.z + 1)
        kal_obj.measure(np.asarray([x, y]).reshape(2, 1), gnss_ts)
        # 终端打印卡尔曼滤波GNSS校正
        print(f"卡尔曼校正: GNSS位置({x:.2f}, {y:.2f}) 校正后位置({kal_obj.state[0,0]:.2f}, {kal_obj.state[1,0]:.2f}) 时间戳{gnss_ts:.2f}")

a.imu_reg_callback(imu_kal_listener)
a.gnss_reg_callback(gnss_kal_listener)

# -------------------------- 数据采集等待 --------------------------
print(f"开始采集数据，时长：{save_time}秒")
time.sleep(save_time)
print(f"数据采集结束！IMU数据：{len(imu_list)}/{imu_len}，GNSS数据：{len(gnss_list)}/{gnss_len}")

# -------------------------- 数据保存 --------------------------
with open('imu_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'timestamp'])
    for data in imu_list:
        accel, gyro, ts = data
        writer.writerow([accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2], ts])

with open('gnss_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y', 'z', 'timestamp'])
    for data in gnss_list:
        pos, ts = data
        writer.writerow([pos[0], pos[1], pos[2], ts])

print("数据已保存到 imu_data.csv 和 gnss_data.csv")

# -------------------------- 可视化（与原始逻辑完全一致） --------------------------
# GNSS与真实位置对比
if len(gnss_list) > 0 and len(kal_rpos_list) > 0:
    rpos_xy = np.asarray([(x[0][0], x[0][1]) for x in kal_rpos_list])
    gnss_xy = np.asarray([(x[0][0], x[0][1]) for x in kal_gnss_list])
    plt.figure(figsize=(8, 6))
    plt.scatter(-gnss_xy[:, 0], gnss_xy[:, 1], 0.3, label="GNSS data", color='red')
    plt.plot(-rpos_xy[:, 0], rpos_xy[:, 1], label="Real position")
    plt.legend()
    plt.show()

# IMU加速度时序
if len(imu_list) > 0:
    plt.plot([max(x[0][0],-10) for x in imu_list])
    plt.title('x axis imu acceleration')
    plt.show()
    plt.plot([max(x[0][1], -10) for x in imu_list])
    plt.title('y axis imu acceleration')
    plt.show()
    plt.plot([max(x[0][2], -10) for x in imu_list])
    plt.title('z axis imu acceleration')
    plt.show()

# 纯IMU积分：X轴速度对比
if len(int_rvel_list) > 0 and len(int_obj.states) > 0:
    plt.plot([x[0][0] for x in int_rvel_list], label='actual')
    plt.plot([x[0][3,0] for x in int_obj.states], label='predicted')
    plt.title('x axis velocity')
    plt.legend()
    plt.show()

# 纯IMU积分：Y轴速度对比
if len(int_rvel_list) > 0 and len(int_obj.states) > 0:
    plt.plot([x[0][1] for x in int_rvel_list], label='actual')
    plt.plot([x[0][4,0] for x in int_obj.states], label='predicted')
    plt.title('y axis velocity')
    plt.legend()
    plt.show()

# 纯IMU积分：偏航角对比
if len(int_ryaw_list) > 0 and len(int_obj.states) > 0:
    plt.plot([x[0] for x in int_ryaw_list], label='actual')
    plt.plot([x[0][2,0] for x in int_obj.states], label='predicted')
    plt.title('yaw')
    plt.legend()
    plt.show()

# 纯IMU积分：位置轨迹
if len(int_rpos_list) > 0 and len(int_obj.states) > 0:
    rpos_xy = np.asarray([(x[0][0], x[0][1]) for x in int_rpos_list])
    gnss_xy = np.asarray([(x[0][0], x[0][1]) for x in int_obj.states])
    plt.plot(-gnss_xy[:,0], gnss_xy[:,1], label="IMU prediction")
    plt.plot(-rpos_xy[:,0], rpos_xy[:,1], label="Real position")
    plt.legend()
    plt.show()

print("Time : %.2f to %.2f"%(int_rpos_list[0][1], int_rpos_list[-1][1]))
print("Time : %.2f to %.2f"%(int_obj.states[0][1], int_obj.states[-1][1]))

# 卡尔曼滤波：X轴速度对比
if len(kal_rvel_list) > 0 and len(kal_obj.states) > 0:
    plt.plot([x[0][0] for x in kal_rvel_list], label='actual')
    plt.plot([x[0][3,0] for x in kal_obj.states if not x[2]], label='predicted')
    plt.title('x axis velocity')
    plt.legend()
    plt.show()

# 卡尔曼滤波：Y轴速度对比
if len(kal_rvel_list) > 0 and len(kal_obj.states) > 0:
    plt.plot([x[0][1] for x in kal_rvel_list], label='actual')
    plt.plot([x[0][4,0] for x in kal_obj.states if not x[2]], label='predicted')
    plt.title('y axis velocity')
    plt.legend()
    plt.show()

# 卡尔曼滤波：偏航角对比
if len(kal_ryaw_list) > 0 and len(kal_obj.states) > 0:
    plt.plot([x[0] for x in kal_ryaw_list], label='actual')
    plt.plot([x[0][2,0] for x in kal_obj.states if not x[2]], label='predicted')
    plt.title('yaw')
    plt.legend()
    plt.show()

# 卡尔曼滤波：位置轨迹（修复维度错误）
if len(kal_rpos_list) > 0 and len(kal_obj.states) > 0:
    k_rpos_xy = np.asarray([(x[0][0], x[0][1]) for x in kal_rpos_list])
    k_pred_xy = np.asarray([(x[0][0], x[0][1]) for x in kal_obj.states if not x[2]])
    k_gnss_xy = np.asarray([(x[0][0], x[0][1]) for x in kal_gnss_list])
    plt.plot(-k_pred_xy[:,0], k_pred_xy[:,1], label="Kalman filter prediction", color='orange')  # 修复维度索引
    plt.plot(-k_rpos_xy[:,0], k_rpos_xy[:,1], label="Real position", color='green')
    plt.scatter(-k_gnss_xy[:,0], k_gnss_xy[:,1], 0.3, label="GNSS data", color='red')
    plt.legend()
    plt.show()

print("Time : %.2f to %.2f"%(kal_rpos_list[0][1], kal_rpos_list[-1][1]))
print("Time : %.2f to %.2f"%(kal_obj.states[0][1], kal_obj.states[-1][1]))

print("所有可视化图表生成完成！")
