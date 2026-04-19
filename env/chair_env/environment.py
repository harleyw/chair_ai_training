# MIT License
#
# Copyright (c) 2026 Harley Wang (王华)
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

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import os
import logging
from env.chair_model import ErgonomicChair
from env.human_model.human_model import HumanModel
from env.sensors.sensors import PressureSensorArray, PostureSensor, Timer

logger = logging.getLogger(__name__)


class ErgonomicChairEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        self.physics_client = None
        self.direct_mode = (render_mode is None)
        
        self.chair = None
        self.human = None
        self.pressure_sensor = None
        self.posture_sensor = None
        self.timer = None
        
        self.user_weight = 70.0
        self.user_height = 1.70
        self.user_body_type = 0.5
        
        self.simulation_step = 0
        self.max_steps = 1000
        self.time_step = 1.0
        
        self.episode_reward = 0.0
        
        self.w_comfort = 1.0
        self.w_pressure = 0.8
        self.w_static = 0.5
        self.w_energy = 0.3
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )
        
        self._init_simulation()
    
    def _init_simulation(self):
        if self.direct_mode:
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI)
        
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=10,
            physicsClientId=self.physics_client
        )
        
        self._load_plane()
    
    def _load_plane(self):
        pybullet_data_dirs = [
            os.path.join(os.path.dirname(p.__file__), 'examples', 'pybullet', 'gym', 'pybullet', 'envs'),
            os.path.join(os.path.dirname(p.__file__), 'data'),
        ]
        
        for data_dir in pybullet_data_dirs:
            plane_path = os.path.join(data_dir, 'plane.urdf')
            if os.path.exists(plane_path):
                try:
                    p.loadURDF(plane_path, [0, 0, 0], physicsClientId=self.physics_client)
                    return
                except Exception as e:
                    logger.debug(f"Failed to load plane from {plane_path}: {e}")
                    continue
        
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[10, 10, 0.1],
            physicsClientId=self.physics_client
        )
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[10, 10, 0.1],
            rgbaColor=[0.2, 0.2, 0.2, 1],
            physicsClientId=self.physics_client
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[0, 0, -0.1],
            physicsClientId=self.physics_client
        )
    
    def _create_user(self, height=None, weight=None, body_type=None):
        if height is not None:
            self.user_height = height
        if weight is not None:
            self.user_weight = weight
        if body_type is not None:
            self.user_body_type = body_type
        
        self.human = HumanModel(
            physics_client_id=self.physics_client,
            height=self.user_height,
            weight=self.user_weight,
            body_type=self.user_body_type
        )
        self.human.build()
    
    def _init_sensors(self):
        self.pressure_sensor = PressureSensorArray(rows=8, cols=8)
        self.posture_sensor = PostureSensor()
        self.timer = Timer()
    
    def _get_contact_points(self):
        contact_points = []
        
        if self.chair is None or self.human is None:
            return contact_points
        
        contacts = p.getContactPoints(physicsClientId=self.physics_client)
        
        for contact in contacts:
            body_a = contact[1]
            body_b = contact[2]
            
            if (body_a == self.human.body_id and body_b in self.chair.chair_parts.values()) or \
               (body_b == self.human.body_id and body_a in self.chair.chair_parts.values()):
                
                contact_info = {
                    'position': contact[5],
                    'normal_force': contact[9]
                }
                contact_points.append(contact_info)
        
        return contact_points
    
    def _get_observation(self):
        chair_state = self.chair.get_state()
        
        human_state = self.human.get_body_state()
        if human_state is not None:
            posture_angles = np.array([
                human_state['orientation'][0],
                human_state['orientation'][1],
                human_state['orientation'][2]
            ])
        else:
            posture_angles = np.zeros(3)
        
        posture_sensor_state = self.posture_sensor.get_state()
        if posture_sensor_state is not None:
            posture_info = posture_sensor_state[:3]
        else:
            posture_info = np.zeros(3)
        
        current_duration = self.timer.get_current_posture_duration()
        time_info = np.array([current_duration])
        
        user_info = np.array([self.user_weight, self.user_body_type])
        
        fatigue_info = np.array([self.human.get_fatigue_factor()])
        
        pressure_readings = self.pressure_sensor.get_flattened_readings()
        pressure_avg = np.array([self.pressure_sensor.get_average_pressure()])
        pressure_max = np.array([self.pressure_sensor.get_max_pressure()])
        
        obs = np.concatenate([
            chair_state,
            posture_angles,
            posture_info,
            time_info,
            user_info,
            fatigue_info,
            pressure_avg,
            pressure_max
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self, action_magnitude):
        posture_type = self.human.get_posture_type()
        
        spine_curvature_diff = self.human.calculate_spine_curvature()
        
        ideal_spine_curvature = 0.0
        spine_alignment_reward = np.exp(-2.0 * spine_curvature_diff ** 2)
        
        pressure_variance = self.pressure_sensor.get_pressure_variance()
        pressure_uniformity_reward = np.exp(-5.0 * pressure_variance)
        
        comfort_reward = 0.5 * spine_alignment_reward + 0.5 * pressure_uniformity_reward
        
        max_pressure = self.pressure_sensor.get_max_pressure()
        pressure_threshold = 50.0
        if max_pressure > pressure_threshold:
            pressure_penalty = (max_pressure - pressure_threshold) / pressure_threshold
        else:
            pressure_penalty = 0.0
        
        static_duration = self.timer.get_current_posture_duration()
        static_threshold = 900.0
        if static_duration > static_threshold:
            static_penalty = (static_duration - static_threshold) / static_threshold
        else:
            static_penalty = 0.0
        
        energy_penalty = action_magnitude * 0.01
        
        total_reward = (
            self.w_comfort * comfort_reward
            - self.w_pressure * pressure_penalty
            - self.w_static * static_penalty
            - self.w_energy * energy_penalty
        )
        
        return total_reward
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        action_magnitude = self.chair.apply_action(action)
        
        self.human.update_fatigue(self.time_step)
        self.human.update_center_of_mass()
        
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        contact_points = self._get_contact_points()
        pressure_modifier = self.human.get_pressure_distribution_modifier()
        self._apply_pressure_modification(contact_points, pressure_modifier)
        self.pressure_sensor.simulate_reading(contact_points)
        
        human_state = self.human.get_body_state()
        self.posture_sensor.update(human_state)
        
        posture_type = self.human.get_posture_type()
        self.timer.update(posture_type, self.time_step)
        
        reward = self._calculate_reward(action_magnitude)
        
        self.episode_reward += reward
        self.simulation_step += 1
        
        obs = self._get_observation()
        
        terminated = False
        truncated = (self.simulation_step >= self.max_steps)
        
        info = {
            'posture_type': posture_type,
            'current_reward': reward,
            'episode_reward': self.episode_reward,
            'step': self.simulation_step,
            'fatigue_level': self.human.get_fatigue_factor(),
            'body_type': self.user_body_type
        }
        
        return obs, reward, terminated, truncated, info
    
    def _apply_pressure_modification(self, contact_points, modifier):
        for contact in contact_points:
            contact['normal_force'] *= modifier['fatigue_modifier']
            contact['normal_force'] *= modifier['contact_area_factor']
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.chair is not None:
            self.chair.remove()
        if self.human is not None:
            self.human.remove()
        
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=10,
            physicsClientId=self.physics_client
        )
        
        self._load_plane()
        
        self.chair = ErgonomicChair(physics_client_id=self.physics_client)
        self.chair.build()
        
        if options is not None:
            self._create_user(
                height=options.get('height', 1.70),
                weight=options.get('weight', 70.0),
                body_type=options.get('body_type', 0.5)
            )
        else:
            self._create_user()
        
        self._init_sensors()
        
        self.simulation_step = 0
        self.episode_reward = 0.0
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def render(self):
        pass
    
    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
