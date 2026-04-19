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

import pybullet as p
import numpy as np


class HumanModel:
    def __init__(self, physics_client_id=0, height=1.70, weight=70, body_type=0.5):
        self.physics_client_id = physics_client_id
        self.height = height
        self.weight = weight
        self.body_type = body_type
        
        self.bmi = weight / (height ** 2)
        self.thigh_length = height * 0.245
        self.torso_length = height * 0.295
        self.lower_leg_length = height * 0.245
        self.head_length = height * 0.125
        self.upper_arm_length = height * 0.185
        self.lower_arm_length = height * 0.165
        
        self.fat_ratio = 0.2 + body_type * 0.3
        
        self.body_id = None
        self.joint_indices = {}
        self.body_parts = {}
        
        self.posture_type = 'neutral'
        
        self.fatigue_level = 0.0
        self.fatigue_rate = 0.0005
        self.max_fatigue = 1.0
        
        self.center_of_mass_offset = np.array([0.0, 0.0, 0.0])
        self.com_shift_frequency = 0.01
        self.com_shift_magnitude = 0.02
        self.simulation_step = 0
        
        self.micro_movement_prob = 0.05
        self.micro_movement_magnitude = 0.01
        
    def build(self, base_position=[0, 0.0, 0.5]):
        torso_height = 0.35
        head_radius = 0.1
        thigh_length = self.thigh_length
        lower_leg_length = self.lower_leg_length
        
        total_mass = self.weight
        torso_mass = total_mass * 0.50
        head_mass = total_mass * 0.07
        thigh_mass = total_mass * 0.10
        lower_leg_mass = total_mass * 0.06
        
        width_factor = 1.0 + self.fat_ratio * 0.3
        torso_width = 0.12 * width_factor
        torso_depth = 0.10 * width_factor
        thigh_width = 0.08 * width_factor
        lower_leg_width = 0.07 * width_factor
        
        link_Masses = [
            head_mass,
            thigh_mass,
            thigh_mass,
            lower_leg_mass,
            lower_leg_mass
        ]
        
        linkCollisionShapeIndices = [
            p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=head_radius,
                physicsClientId=self.physics_client_id
            ),
            p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[thigh_width, thigh_length / 2, thigh_width],
                physicsClientId=self.physics_client_id
            ),
            p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[thigh_width, thigh_length / 2, thigh_width],
                physicsClientId=self.physics_client_id
            ),
            p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[lower_leg_width, lower_leg_length / 2, lower_leg_width],
                physicsClientId=self.physics_client_id
            ),
            p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[lower_leg_width, lower_leg_length / 2, lower_leg_width],
                physicsClientId=self.physics_client_id
            )
        ]
        
        linkVisualShapeIndices = [
            p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=head_radius,
                rgbaColor=[0.8, 0.5, 0.3, 0.9],
                physicsClientId=self.physics_client_id
            ),
            p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[thigh_width, thigh_length / 2, thigh_width],
                rgbaColor=[0.65, 0.45, 0.3, 0.9],
                physicsClientId=self.physics_client_id
            ),
            p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[thigh_width, thigh_length / 2, thigh_width],
                rgbaColor=[0.65, 0.45, 0.3, 0.9],
                physicsClientId=self.physics_client_id
            ),
            p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[lower_leg_width, lower_leg_length / 2, lower_leg_width],
                rgbaColor=[0.65, 0.45, 0.3, 0.9],
                physicsClientId=self.physics_client_id
            ),
            p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[lower_leg_width, lower_leg_length / 2, lower_leg_width],
                rgbaColor=[0.65, 0.45, 0.3, 0.9],
                physicsClientId=self.physics_client_id
            )
        ]
        
        linkPositions = [
            [0, 0, torso_height / 2 + head_radius],
            [0.08 * width_factor, 0, -torso_height / 2],
            [-0.08 * width_factor, 0, -torso_height / 2],
            [0.08 * width_factor, 0, -torso_height / 2 - thigh_length / 2],
            [-0.08 * width_factor, 0, -torso_height / 2 - thigh_length / 2]
        ]
        
        linkOrientations = [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ]
        
        linkInertialFramePositions = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        
        linkInertialFrameOrientations = [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ]
        
        parentIndices = [-1, -1, -1, 1, 2]
        
        jointTypes = [
            p.JOINT_FIXED,
            p.JOINT_FIXED,
            p.JOINT_FIXED,
            p.JOINT_FIXED,
            p.JOINT_FIXED
        ]
        
        jointAxis = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]
        
        torso_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[torso_width, torso_depth, torso_height / 2],
            physicsClientId=self.physics_client_id
        )
        
        torso_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[torso_width, torso_depth, torso_height / 2],
            rgbaColor=[0.7, 0.5, 0.3, 0.9],
            physicsClientId=self.physics_client_id
        )
        
        self.body_id = p.createMultiBody(
            baseMass=torso_mass,
            baseCollisionShapeIndex=torso_collision,
            baseVisualShapeIndex=torso_visual,
            basePosition=base_position,
            baseOrientation=[0, 0, 0, 1],
            linkMasses=link_Masses,
            linkCollisionShapeIndices=linkCollisionShapeIndices,
            linkVisualShapeIndices=linkVisualShapeIndices,
            linkPositions=linkPositions,
            linkOrientations=linkOrientations,
            linkInertialFramePositions=linkInertialFramePositions,
            linkInertialFrameOrientations=linkInertialFrameOrientations,
            linkParentIndices=parentIndices,
            linkJointTypes=jointTypes,
            linkJointAxis=jointAxis,
            physicsClientId=self.physics_client_id
        )
        
        self.torso_height = torso_height
        self.torso_width = torso_width
        self.torso_depth = torso_depth
        
        self.body_parts = {
            'torso': self.body_id,
            'torso_height': torso_height,
            'torso_width': torso_width,
            'torso_depth': torso_depth
        }
        
        return self.body_id
    
    def update_fatigue(self, time_elapsed):
        self.fatigue_level = min(self.fatigue_level + self.fatigue_rate * time_elapsed, self.max_fatigue)
        
    def update_center_of_mass(self):
        self.simulation_step += 1
        
        shift_phase = self.simulation_step * self.com_shift_frequency
        self.center_of_mass_offset[0] = self.com_shift_magnitude * np.sin(shift_phase)
        self.center_of_mass_offset[1] = self.com_shift_magnitude * 0.5 * np.cos(shift_phase * 0.7)
        
        if np.random.random() < self.micro_movement_prob:
            self.center_of_mass_offset[0] += np.random.uniform(-self.micro_movement_magnitude, self.micro_movement_magnitude)
            self.center_of_mass_offset[1] += np.random.uniform(-self.micro_movement_magnitude, self.micro_movement_magnitude)
        
        self.center_of_mass_offset *= 0.95
        
    def get_fatigue_factor(self):
        return self.fatigue_level
    
    def get_pressure_distribution_modifier(self):
        pressure_variance_base = 0.1
        
        fatigue_modifier = 1.0 + self.fatigue_level * 0.5
        
        body_width_factor = 1.0 + self.fat_ratio * 0.3
        
        contact_area_factor = 1.0 / (body_width_factor ** 2)
        
        com_offset_magnitude = np.linalg.norm(self.center_of_mass_offset)
        com_pressure_shift = com_offset_magnitude * 2.0
        
        return {
            'variance_base': pressure_variance_base,
            'fatigue_modifier': fatigue_modifier,
            'contact_area_factor': contact_area_factor,
            'com_pressure_shift': com_pressure_shift
        }
    
    def get_body_state(self):
        if self.body_id is None:
            return None
        
        pos, orn = p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.physics_client_id
        )
        
        vel, ang_vel = p.getBaseVelocity(
            self.body_id,
            physicsClientId=self.physics_client_id
        )
        
        torso_pos = np.array(pos) + self.center_of_mass_offset
        torso_vel = np.array(vel)
        
        euler = p.getEulerFromQuaternion(orn)
        
        fatigue_jitter = self.fatigue_level * 0.02 * np.random.randn(3)
        torso_angle = np.array(euler) + fatigue_jitter
        
        return {
            'position': torso_pos,
            'orientation': torso_angle,
            'velocity': torso_vel,
            'angular_velocity': np.array(ang_vel)
        }
    
    def calculate_spine_curvature(self):
        if self.body_id is None:
            return 0.0
        
        pos, orn = p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.physics_client_id
        )
        
        euler = p.getEulerFromQuaternion(orn)
        
        fatigue_postural_drift = self.fatigue_level * 0.05
        
        ideal_curvature = fatigue_postural_drift
        actual_curvature = euler[0]
        
        curvature_diff = abs(actual_curvature - ideal_curvature)
        
        return curvature_diff
    
    def get_posture_type(self):
        if self.body_id is None:
            return 'unknown'
        
        pos, orn = p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.physics_client_id
        )
        
        euler = p.getEulerFromQuaternion(orn)
        
        pitch = euler[0]
        roll = euler[1]
        
        fatigue_threshold_increase = self.fatigue_level * 0.05
        base_threshold = 0.1
        
        effective_threshold = base_threshold + fatigue_threshold_increase
        
        if abs(pitch) < effective_threshold and abs(roll) < effective_threshold:
            self.posture_type = 'neutral'
        elif pitch > effective_threshold:
            self.posture_type = 'leaning_forward'
        elif pitch < -effective_threshold:
            self.posture_type = 'leaning_backward'
        elif roll > effective_threshold:
            self.posture_type = 'leaning_left'
        elif roll < -effective_threshold:
            self.posture_type = 'leaning_right'
        
        return self.posture_type
    
    def reset(self, base_position=[0, 0.0, 0.5]):
        if self.body_id is not None:
            p.resetBasePositionAndOrientation(
                self.body_id,
                base_position,
                [0, 0, 0, 1],
                physicsClientId=self.physics_client_id
            )
            
            p.resetBaseVelocity(
                self.body_id,
                [0, 0, 0],
                [0, 0, 0],
                physicsClientId=self.physics_client_id
            )
        
        self.fatigue_level = 0.0
        self.center_of_mass_offset = np.array([0.0, 0.0, 0.0])
        self.simulation_step = 0
        self.posture_type = 'neutral'
    
    def remove(self):
        if self.body_id is not None:
            p.removeBody(self.body_id, physicsClientId=self.physics_client_id)
