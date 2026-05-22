#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2021 Ashwin Rajesh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import carla
import numpy as np
import time


def deg_to_rad(val):
    return val * np.pi / 180


class agent:
    # Constructor
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.world  = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()

        self.vehicle = None
        self.actor_list = []

        map_geo = self.world.get_map().transform_to_geolocation(carla.Location(0,0,0))
        
        self.geo_centre_lat = deg_to_rad(map_geo.latitude) 
        self.geo_centre_lon = deg_to_rad(map_geo.longitude)
        self.geo_centre_alt = map_geo.altitude
        print("✅ CARLA 连接成功！世界地图名称：", self.world.get_map().name)


    # Spawn the vehicle randomly
    def spawn_vehicle(self, index=None):
        # Create blueprint and transform
        car_bp = np.random.choice(self.bp_lib.filter("vehicle.*.*"))
        print(" Vehicle model : %s %s"%(car_bp.id.split('.')[1], car_bp.id.split('.')[2]))
        if(index == None):    
            car_tf = np.random.choice(self.world.get_map().get_spawn_points())
            print(" Spawning vehicle at location : %d, %d, %d (random generation)"%(car_tf.location.x, car_tf.location.y, car_tf.location.z))
        else:
            car_tf = self.world.get_map().get_spawn_points()[index]
            print(" Spawning vehicle at location : %d, %d, %d (spawn point no. %d)"%(car_tf.location.x, car_tf.location.y, car_tf.location.z, index))

        # Spawn vehicle, append to actor list and set autopilot
        self.vehicle = self.world.spawn_actor(car_bp, car_tf)
        if(self.vehicle == None):
            print(" Vehicle could not be spawned")
            return
        self.actor_list.append(self.vehicle)
        self.vehicle.set_autopilot(True)

    # 生成 1 个 IMU 传感器
    def spawn_imu(self, period=0.1, accel_std_dev = 0, gyro_std_dev = 0):
        # Check if the vehicle was spawned
        if(self.vehicle == None):
            print(" Error : spawn car first")
            return

        # Create blueprint and transform
        imu_bp = self.bp_lib.find("sensor.other.imu")
        imu_bp.set_attribute('sensor_tick', '0.1')
        imu_bp.set_attribute('noise_gyro_stddev_y',  str(gyro_std_dev))
        imu_bp.set_attribute('noise_gyro_stddev_x',  str(gyro_std_dev))
        imu_bp.set_attribute('noise_gyro_stddev_z',  str(gyro_std_dev))
        imu_bp.set_attribute('noise_accel_stddev_y', str(accel_std_dev))
        imu_bp.set_attribute('noise_accel_stddev_x', str(accel_std_dev))
        imu_bp.set_attribute('noise_accel_stddev_z', str(accel_std_dev))
        imu_tf = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))

        # Spawning the sensor and appending to list
        self.imu = self.world.spawn_actor(imu_bp, imu_tf, attach_to=self.vehicle)
        self.actor_list.append(self.imu)
        # For timing
        self.imu_time = 0
        self.imu_per  = period
        # Register listen function
        self.imu_callbacks = []
        self.imu.listen(self.imu_listen)

    # Listener function for IMU sensor
    def imu_listen(self, data):
        if(data.timestamp - self.imu_time < self.imu_per):
            return
        self.imu_time = data.timestamp
        self.imu_data = data

        for c in self.imu_callbacks:
            c(data)

    # Register a function to be called when IMU data is received
    def imu_reg_callback(self, callback):
        self.imu_callbacks.append(callback)

    # Convert from geographic (lat, lon, alt) in data to x, y and z coordinates
    def gnss_to_xyz(self, data):
        
        lat = deg_to_rad(data.latitude)
        lon = deg_to_rad(data.longitude)
        alt = data.altitude 
        
        rad_y  = 6.357e6
        rad_x  = 6.378e6

        x = (lon - self.geo_centre_lon) * np.cos(self.geo_centre_lat) * rad_x
        y = (self.geo_centre_lat - lat) * rad_y
        z = alt - self.geo_centre_alt

        return [x, y, z]

    # Spawn a GNSS sensor
    def spawn_gnss(self, period=0.1, std_dev=0.1):
        # Check if the vehicle was spawned
        if(self.vehicle == None):
            print(" Error : spawn car first")
            return

        # Define the blueprint and transform
        gnss_bp = self.bp_lib.find("sensor.other.gnss")
        gnss_bp.set_attribute('sensor_tick', '0.1')
        gnss_bp.set_attribute('noise_lat_stddev', str(std_dev))
        gnss_bp.set_attribute('noise_lon_stddev', str(std_dev))
        gnss_tf = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))
        
        # Spawning the sensor and appending to list
        self.gnss = self.world.spawn_actor(gnss_bp, gnss_tf, attach_to=self.vehicle)
        self.actor_list.append(self.gnss)
        # For timing
        self.gnss_time = 0
        self.gnss_per  = period
        # Register listen callback
        self.gnss_callbacks = []
        self.gnss.listen(self.gnss_listen)

    # Listener function for GNSS sensor
    def gnss_listen(self, data):
        if(data.timestamp - self.gnss_time < self.gnss_per):
            return
        self.gnss_time = data.timestamp
        self.gnss_data = data

        for c in self.gnss_callbacks:
            c(data)

    # Register a function to be called when gnss data is received
    def gnss_reg_callback(self, callback):
        self.gnss_callbacks.append(callback)
        
    # Destructor
    def __del__(self):
        for a in list(self.actor_list):
            a.destroy()
            self.actor_list.remove(a)

    # Destroy all actors
    def destroy_actors(self):
        for a in list(self.actor_list):
            a.destroy()
            self.actor_list.remove(a)


if __name__ == "__main__":
    carla_agent = agent()

