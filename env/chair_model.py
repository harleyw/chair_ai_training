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


class ErgonomicChair:
    def __init__(self, physics_client_id=0):
        self.physics_client_id = physics_client_id
        
        self.seat_height = 0.45
        self.backrest_angle = np.pi / 2
        self.lumbar_support_pos = 0.0
        self.lumbar_support_thickness = 0.05
        self.headrest_height = 0.0
        self.headrest_angle = 0.0
        self.armrest_left_height = 0.0
        self.armrest_right_height = 0.0
        
        self.min_seat_height = 0.40
        self.max_seat_height = 0.55
        
        self.min_backrest_angle = np.pi / 3
        self.max_backrest_angle = np.pi / 2.2
        
        self.min_lumbar_pos = -0.1
        self.max_lumbar_pos = 0.1
        
        self.min_lumbar_thickness = 0.02
        self.max_lumbar_thickness = 0.1
        
        self.min_headrest_height = 0.0
        self.max_headrest_height = 0.15
        
        self.min_headrest_angle = -np.pi / 6
        self.max_headrest_angle = np.pi / 6
        
        self.min_armrest_height = 0.0
        self.max_armrest_height = 0.15
        
        self.chair_base_id = None
        self.joint_indices = {}
        
    def build(self, base_position=[0, 0, 0]):
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.05],
            physicsClientId=self.physics_client_id
        )
        
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.05],
            rgbaColor=[0.3, 0.3, 0.3, 1],
            physicsClientId=self.physics_client_id
        )
        
        self.chair_base_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=base_position,
            physicsClientId=self.physics_client_id
        )
        
        seat_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.22, 0.22, 0.06],
            physicsClientId=self.physics_client_id
        )
        
        seat_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.22, 0.22, 0.06],
            rgbaColor=[0.2, 0.2, 0.25, 1],
            physicsClientId=self.physics_client_id
        )
        
        seat_link = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=seat_collision,
            baseVisualShapeIndex=seat_visual,
            basePosition=[0, 0, self.seat_height],
            physicsClientId=self.physics_client_id
        )
        
        backrest_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.22, 0.06, 0.35],
            physicsClientId=self.physics_client_id
        )
        
        backrest_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.22, 0.06, 0.35],
            rgbaColor=[0.2, 0.2, 0.25, 1],
            physicsClientId=self.physics_client_id
        )
        
        backrest_link = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=backrest_collision,
            baseVisualShapeIndex=backrest_visual,
            basePosition=[0, -0.2, self.seat_height + 0.35],
            physicsClientId=self.physics_client_id
        )
        
        lumbar_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.15, self.lumbar_support_thickness, 0.08],
            physicsClientId=self.physics_client_id
        )
        
        lumbar_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.15, self.lumbar_support_thickness, 0.08],
            rgbaColor=[0.25, 0.25, 0.3, 1],
            physicsClientId=self.physics_client_id
        )
        
        lumbar_link = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=lumbar_collision,
            baseVisualShapeIndex=lumbar_visual,
            basePosition=[0, -0.15, self.seat_height + 0.15 + self.lumbar_support_pos],
            physicsClientId=self.physics_client_id
        )
        
        headrest_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.12, 0.05, 0.1],
            physicsClientId=self.physics_client_id
        )
        
        headrest_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.12, 0.05, 0.1],
            rgbaColor=[0.25, 0.25, 0.3, 1],
            physicsClientId=self.physics_client_id
        )
        
        headrest_link = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=headrest_collision,
            baseVisualShapeIndex=headrest_visual,
            basePosition=[0, -0.18, self.seat_height + 0.7 + self.headrest_height],
            physicsClientId=self.physics_client_id
        )
        
        self.chair_parts = {
            'seat': seat_link,
            'backrest': backrest_link,
            'lumbar': lumbar_link,
            'headrest': headrest_link
        }
        
        return self.chair_base_id
    
    def get_state(self):
        return np.array([
            self.seat_height,
            self.backrest_angle,
            self.lumbar_support_pos,
            self.lumbar_support_thickness,
            self.headrest_height,
            self.headrest_angle,
            self.armrest_left_height,
            self.armrest_right_height
        ])
    
    def apply_action(self, action):
        seat_height_delta = action[0] * 0.005
        backrest_angle_delta = action[1] * 0.01
        lumbar_pos_delta = action[2] * 0.005
        lumbar_thickness_delta = action[3] * 0.002
        headrest_height_delta = action[4] * 0.005
        headrest_angle_delta = action[5] * 0.01
        armrest_left_delta = action[6] * 0.005
        armrest_right_delta = action[7] * 0.005
        
        self.seat_height = np.clip(
            self.seat_height + seat_height_delta,
            self.min_seat_height,
            self.max_seat_height
        )
        
        self.backrest_angle = np.clip(
            self.backrest_angle + backrest_angle_delta,
            self.min_backrest_angle,
            self.max_backrest_angle
        )
        
        self.lumbar_support_pos = np.clip(
            self.lumbar_support_pos + lumbar_pos_delta,
            self.min_lumbar_pos,
            self.max_lumbar_pos
        )
        
        self.lumbar_support_thickness = np.clip(
            self.lumbar_support_thickness + lumbar_thickness_delta,
            self.min_lumbar_thickness,
            self.max_lumbar_thickness
        )
        
        self.headrest_height = np.clip(
            self.headrest_height + headrest_height_delta,
            self.min_headrest_height,
            self.max_headrest_height
        )
        
        self.headrest_angle = np.clip(
            self.headrest_angle + headrest_angle_delta,
            self.min_headrest_angle,
            self.max_headrest_angle
        )
        
        self.armrest_left_height = np.clip(
            self.armrest_left_height + armrest_left_delta,
            self.min_armrest_height,
            self.max_armrest_height
        )
        
        self.armrest_right_height = np.clip(
            self.armrest_right_height + armrest_right_delta,
            self.min_armrest_height,
            self.max_armrest_height
        )
        
        self._update_positions()
        
        action_magnitude = np.sum(np.abs(action))
        return action_magnitude
    
    def _update_positions(self):
        if self.chair_parts is None:
            return
        
        p.resetBasePositionAndOrientation(
            self.chair_parts['seat'],
            [0, 0, self.seat_height],
            [0, 0, 0, 1],
            physicsClientId=self.physics_client_id
        )
        
        backrest_z = self.seat_height + 0.35
        p.resetBasePositionAndOrientation(
            self.chair_parts['backrest'],
            [0, -0.2, backrest_z],
            [0, 0, 0, 1],
            physicsClientId=self.physics_client_id
        )
        
        lumbar_z = self.seat_height + 0.15 + self.lumbar_support_pos
        lumbar_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.15, self.lumbar_support_thickness, 0.08],
            physicsClientId=self.physics_client_id
        )
        p.changeVisualShape(
            self.chair_parts['lumbar'],
            -1,
            rgbaColor=[0.25, 0.25, 0.3, 1],
            physicsClientId=self.physics_client_id
        )
        p.resetBasePositionAndOrientation(
            self.chair_parts['lumbar'],
            [0, -0.15, lumbar_z],
            [0, 0, 0, 1],
            physicsClientId=self.physics_client_id
        )
        
        headrest_z = self.seat_height + 0.7 + self.headrest_height
        p.resetBasePositionAndOrientation(
            self.chair_parts['headrest'],
            [0, -0.18, headrest_z],
            [0, 0, 0, 1],
            physicsClientId=self.physics_client_id
        )
    
    def reset(self):
        self.seat_height = 0.45
        self.backrest_angle = np.pi / 2
        self.lumbar_support_pos = 0.0
        self.lumbar_support_thickness = 0.05
        self.headrest_height = 0.0
        self.headrest_angle = 0.0
        self.armrest_left_height = 0.0
        self.armrest_right_height = 0.0
        
        if self.chair_parts is not None:
            self._update_positions()
    
    def remove(self):
        if self.chair_base_id is not None:
            p.removeBody(self.chair_base_id, physicsClientId=self.physics_client_id)
        if self.chair_parts is not None:
            for part_id in self.chair_parts.values():
                p.removeBody(part_id, physicsClientId=self.physics_client_id)
