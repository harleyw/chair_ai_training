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
from env.chair_env.environment import ErgonomicChairEnv
import numpy as np


def test_environment():
    print("Creating environment...")
    env = ErgonomicChairEnv(render_mode=None)
    
    print("Testing reset...")
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation: {obs[:5]}...")
    
    print("Testing random actions...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step + 1}: Reward = {reward:.4f}, Posture = {info.get('posture_type', 'unknown')}")
        
        if terminated or truncated:
            print("  Episode ended")
            break
    
    print("Testing close...")
    env.close()
    
    print("\nAll environment tests passed!")


if __name__ == '__main__':
    try:
        test_environment()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
