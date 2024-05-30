# simulation_testing.py

"""
Simulation and Testing Module for Autonomous Vehicle Navigation

This module contains the implementation of simulation and testing procedures
for validating the navigation algorithm in simulated environments.

Tools Used:
- CARLA Simulator
- ROS (Robot Operating System)
"""

import carla
import random
import time

class CarlaSimulation:
    def __init__(self, host='localhost', port=2000):
        """
        Initialize the CarlaSimulation class.

        :param host: str, CARLA server host
        :param port: int, CARLA server port
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

    def setup_environment(self):
        """
        Set up the simulation environment in CARLA.
        """
        # Set the weather conditions
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

        # Get the blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # Spawn a vehicle
        vehicle_bp = self.blueprint_library.filter('model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Attach a camera sensor to the vehicle
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        # Attach a LiDAR sensor to the vehicle
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)

        # Attach an IMU sensor to the vehicle
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.imu = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)

    def run_simulation(self, duration=60):
        """
        Run the simulation for a specified duration.

        :param duration: int, duration of the simulation in seconds
        """
        self.vehicle.set_autopilot(True)

        start_time = time.time()
        while time.time() - start_time < duration:
            self.world.tick()
            time.sleep(0.05)

        self.cleanup()

    def cleanup(self):
        """
        Clean up the simulation environment.
        """
        self.camera.destroy()
        self.lidar.destroy()
        self.imu.destroy()
        self.vehicle.destroy()

if __name__ == "__main__":
    sim = CarlaSimulation()
    sim.setup_environment()
    sim.run_simulation(duration=60)
