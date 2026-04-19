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


import sys
import numpy as np
from env.chair_env.environment import ErgonomicChairEnv


def test_dynamic_human_features():
    print("Testing dynamic human features...")
    
    env = ErgonomicChairEnv(render_mode=None)
    
    print("\n1. Testing body type variation:")
    for body_type in [0.2, 0.5, 0.8]:
        obs, info = env.reset(options={'body_type': body_type})
        print(f"   body_type={body_type}: obs shape={obs.shape}, body_type in obs={obs[15]:.1f}")
    
    print("\n2. Testing fatigue accumulation:")
    obs, info = env.reset()
    initial_fatigue = env.human.get_fatigue_factor()
    print(f"   Initial fatigue: {initial_fatigue:.3f}")
    
    for i in range(10):
        action = np.zeros(8)
        obs, reward, terminated, truncated, info = env.step(action)
    
    final_fatigue = env.human.get_fatigue_factor()
    print(f"   After 10 steps: fatigue={final_fatigue:.3f}")
    
    print("\n3. Testing center of mass shift:")
    obs, info = env.reset()
    initial_com = env.human.center_of_mass_offset.copy()
    print(f"   Initial CoM offset: {initial_com}")
    
    for i in range(5):
        env.human.update_center_of_mass()
    
    shifted_com = env.human.center_of_mass_offset.copy()
    print(f"   After 5 shifts: {shifted_com}")
    
    print("\n4. Testing pressure distribution modifier:")
    obs, info = env.reset()
    env.human.fatigue_level = 0.5
    modifier = env.human.get_pressure_distribution_modifier()
    print(f"   Fatigue=0.5: {modifier}")
    
    print("\n5. Testing observation space consistency:")
    print(f"   Expected obs dim: 20, Actual: {obs.shape[0]}")
    assert obs.shape[0] == 20, f"Observation dimension mismatch: {obs.shape[0]} != 20"
    
    env.close()
    
    print("\nAll dynamic human feature tests passed!")


if __name__ == '__main__':
    try:
        test_dynamic_human_features()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
