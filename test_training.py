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
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def create_env(env_idx=0):
    return ErgonomicChairEnv(render_mode=None)


def test_short_training():
    print("Testing environment creation...")
    env = ErgonomicChairEnv(render_mode=None)
    
    print("Testing observation space...")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"  Sample observation shape: {obs.shape}")
    
    assert obs.shape[0] == env.observation_space.shape[0], \
        f"Observation shape mismatch: {obs.shape[0]} != {env.observation_space.shape[0]}"
    
    env.close()
    
    print("\nTesting vector environment creation...")
    vec_env = make_vec_env(create_env, n_envs=2)
    obs = vec_env.reset()
    print(f"  Vector env observation shape: {obs.shape}")
    
    print("\nTesting PPO model creation...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        verbose=1,
        device='cpu'
    )
    print("  PPO model created successfully")
    
    print("\nTraining for 128 steps...")
    model.learn(total_timesteps=128, progress_bar=False)
    print("  Training completed successfully")
    
    print("\nTesting action prediction...")
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"  Predicted action shape: {action.shape}")
    
    vec_env.close()
    
    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    try:
        test_short_training()
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
