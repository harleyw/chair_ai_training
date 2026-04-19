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

import numpy as np
import pybullet as p


class PressureSensorArray:
    def __init__(self, rows=8, cols=8, area_width=0.4, area_height=0.4):
        self.rows = rows
        self.cols = cols
        self.area_width = area_width
        self.area_height = area_height
        
        self.sensor_matrix = np.zeros((rows, cols))
        
    def simulate_reading(self, contact_points):
        self.sensor_matrix = np.zeros((self.rows, self.cols))
        
        for contact in contact_points:
            x, y = contact['position'][:2]
            force = contact['normal_force']
            
            row = int((y + self.area_height / 2) / self.area_height * self.rows)
            col = int((x + self.area_width / 2) / self.area_width * self.cols)
            
            row = np.clip(row, 0, self.rows - 1)
            col = np.clip(col, 0, self.cols - 1)
            
            self.sensor_matrix[row, col] += force
        
        return self.sensor_matrix
    
    def get_pressure_distribution(self):
        return self.sensor_matrix.copy()
    
    def get_average_pressure(self):
        return np.mean(self.sensor_matrix)
    
    def get_max_pressure(self):
        return np.max(self.sensor_matrix)
    
    def get_pressure_variance(self):
        return np.var(self.sensor_matrix)
    
    def get_flattened_readings(self):
        return self.sensor_matrix.flatten()


class PostureSensor:
    def __init__(self):
        self.head_position = np.zeros(3)
        self.shoulder_position = np.zeros(3)
        self.pelvis_position = np.zeros(3)
        
        self.head_angle = 0.0
        self.shoulder_angle = 0.0
        self.pelvis_angle = 0.0
        
    def update(self, body_state):
        if body_state is None:
            return
        
        self.pelvis_position = body_state['position']
        self.pelvis_angle = body_state['orientation'][0]
        
        torso_height = 0.35
        self.shoulder_position = self.pelvis_position + np.array([0, 0, torso_height * 0.6])
        self.shoulder_angle = body_state['orientation'][0]
        
        self.head_position = self.pelvis_position + np.array([0, 0, torso_height * 1.1])
        self.head_angle = body_state['orientation'][0]
        
    def get_state(self):
        return np.concatenate([
            self.head_position,
            self.shoulder_position,
            self.pelvis_position,
            [self.head_angle, self.shoulder_angle, self.pelvis_angle]
        ])
    
    def get_key_angles(self):
        return {
            'head_angle': self.head_angle,
            'shoulder_angle': self.shoulder_angle,
            'pelvis_angle': self.pelvis_angle
        }


class Timer:
    def __init__(self):
        self.current_posture = 'neutral'
        self.posture_start_time = 0.0
        self.current_time = 0.0
        
        self.posture_history = {}
        
    def update(self, posture_type, time_step):
        self.current_time += time_step
        
        if posture_type != self.current_posture:
            duration = self.current_time - self.posture_start_time
            if self.current_posture in self.posture_history:
                self.posture_history[self.current_posture].append(duration)
            
            self.current_posture = posture_type
            self.posture_start_time = self.current_time
        
    def get_current_posture_duration(self):
        return self.current_time - self.posture_start_time
    
    def get_posture_history(self):
        return self.posture_history.copy()
    
    def reset(self, initial_time=0.0):
        self.current_posture = 'neutral'
        self.posture_start_time = initial_time
        self.current_time = initial_time
        self.posture_history = {}
