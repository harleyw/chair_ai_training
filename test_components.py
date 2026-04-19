#!/usr/bin/env python3

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
from env.chair_model import ErgonomicChair
from env.human_model.human_model import HumanModel
from env.sensors.sensors import PressureSensorArray, PostureSensor, Timer


def test_simulation():
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    
    p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=physics_client)
    
    chair = ErgonomicChair(physics_client_id=physics_client)
    chair.build()
    
    human = HumanModel(
        physics_client_id=physics_client,
        height=1.70,
        weight=70.0
    )
    human.build()
    
    pressure_sensor = PressureSensorArray(rows=8, cols=8)
    posture_sensor = PostureSensor()
    timer = Timer()
    
    print("=" * 60)
    print("Simulation Test")
    print("=" * 60)
    
    for i in range(100):
        action = np.random.uniform(-0.5, 0.5, 8)
        chair.apply_action(action)
        
        p.stepSimulation(physicsClientId=physics_client)
        
        human_state = human.get_body_state()
        posture_type = human.get_posture_type()
        
        posture_sensor.update(human_state)
        timer.update(posture_type, 1.0)
        
        if i % 10 == 0:
            print(f"\nStep {i}:")
            print(f"  Posture: {posture_type}")
            print(f"  Chair State: {chair.get_state()}")
    
    print("\nSimulation test completed!")
    print("Close the GUI window to exit...")
    
    try:
        while True:
            p.stepSimulation(physicsClientId=physics_client)
    except KeyboardInterrupt:
        pass
    
    p.disconnect(physics_client)


def test_reward_function():
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    
    p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=physics_client)
    
    chair = ErgonomicChair(physics_client_id=physics_client)
    chair.build()
    
    human = HumanModel(
        physics_client_id=physics_client,
        height=1.70,
        weight=70.0
    )
    human.build()
    
    pressure_sensor = PressureSensorArray(rows=8, cols=8)
    
    print("=" * 60)
    print("Reward Function Test")
    print("=" * 60)
    
    for _ in range(50):
        action = np.random.uniform(-1.0, 1.0, 8)
        chair.apply_action(action)
        
        p.stepSimulation(physicsClientId=physics_client)
        
        posture_type = human.get_posture_type()
        spine_diff = human.calculate_spine_curvature()
        
        max_pressure = pressure_sensor.get_max_pressure()
        pressure_variance = pressure_sensor.get_pressure_variance()
        
        print(f"Posture: {posture_type}")
        print(f"  Spine Curvature Diff: {spine_diff:.4f}")
        print(f"  Max Pressure: {max_pressure:.2f}")
        print(f"  Pressure Variance: {pressure_variance:.4f}")
    
    p.disconnect(physics_client)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test chair simulation components')
    
    parser.add_argument(
        '--test',
        type=str,
        choices=['simulation', 'reward'],
        default='simulation',
        help='Which test to run (default: simulation)'
    )
    
    args = parser.parse_args()
    
    if args.test == 'simulation':
        test_simulation()
    elif args.test == 'reward':
        test_reward_function()
