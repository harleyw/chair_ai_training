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

import os
import time
import numpy as np
from datetime import datetime
import logging
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=1, log_dir='./logs'):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_dir = log_dir
        self.start_time = time.time()
        self.last_log_time = time.time()
        
    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                if self.verbose >= 1 and len(self.episode_rewards) % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    logger.info(
                        f"Episode {len(self.episode_rewards)}: "
                        f"Reward={info['episode']['r']:.2f}, "
                        f"Mean(100)={mean_reward:.2f}, "
                        f"Length={info['episode']['l']}"
                    )
        
        if time.time() - self.last_log_time > 60:
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            logger.info(
                f"Timesteps: {self.num_timesteps:,}, "
                f"FPS: {fps:.0f}, "
                f"Elapsed: {elapsed/60:.1f} min"
            )
            self.last_log_time = time.time()
        
        return True
    
    def plot_rewards(self, save_dir='./logs'):
        if len(self.episode_rewards) == 0:
            logger.warning("No episodes completed yet.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        window = min(100, len(self.episode_rewards))
        rewards = np.array(self.episode_rewards)
        
        if len(rewards) >= window:
            cumsum = np.cumsum(rewards)
            cumsum[window:] = cumsum[window:] - cumsum[:-window]
            rolling_mean = cumsum[window-1:] / window
        
        axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward')
        if len(rewards) >= window:
            axes[0, 0].plot(
                range(window-1, len(rewards)), rolling_mean,
                linewidth=2, label=f'Rolling Mean (window={window})'
            )
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.episode_lengths, alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True)
        
        if len(self.episode_rewards) >= 100:
            axes[1, 0].hist(self.episode_rewards[-1000:], bins=50, alpha=0.7)
            axes[1, 0].set_xlabel('Reward')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Reward Distribution (Last 1000)')
            axes[1, 0].grid(True)
        
        elapsed_minutes = (time.time() - self.start_time) / 60
        axes[1, 1].text(0.1, 0.7, f'Episodes: {len(self.episode_rewards)}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Mean Reward: {np.mean(self.episode_rewards):.2f}', fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'Max Reward: {np.max(self.episode_rewards):.2f}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Elapsed: {elapsed_minutes:.1f} min', fontsize=12)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(save_dir, f'training_progress_{timestamp}.png')
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Training progress saved to {plot_path}")
        plt.close()


class BodyTypeRandomizer(gym.Wrapper):
    def __init__(self, env, height_range=(1.55, 1.85), weight_range=(50, 100), body_type_range=(0.1, 0.9)):
        super().__init__(env)
        self.height_range = height_range
        self.weight_range = weight_range
        self.body_type_range = body_type_range
    
    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        options['height'] = np.random.uniform(*self.height_range)
        options['weight'] = np.random.uniform(*self.weight_range)
        options['body_type'] = np.random.uniform(*self.body_type_range)
        return super().reset(seed=seed, options=options)


def create_env(env_idx=0):
    from env.chair_env.environment import ErgonomicChairEnv
    base_env = ErgonomicChairEnv(render_mode=None)
    return BodyTypeRandomizer(base_env)


def train_ppo(
    total_timesteps=100000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    use_gpu=True,
    log_dir='./logs',
    model_dir='./models',
    load_path=None,
    save_freq=10000
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'run_{timestamp}'
    run_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Ergonomic Chair AI Training")
    logger.info("=" * 60)
    logger.info(f"Run: {run_name}")
    logger.info(f"Total Timesteps: {total_timesteps:,}")
    logger.info(f"Parallel Environments: {n_envs}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"N Steps: {n_steps}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"N Epochs: {n_epochs}")
    logger.info(f"Gamma: {gamma}")
    logger.info(f"GAE Lambda: {gae_lambda}")
    
    device = 'cuda' if (use_gpu and __import__('torch').cuda.is_available()) else 'cpu'
    logger.info(f"Using device: {device}")
    
    vec_env = make_vec_env(
        create_env,
        n_envs=n_envs,
        seed=42
    )
    
    policy_kwargs = dict(
        net_arch=[256, 128, 64],
        activation_fn=__import__('torch').nn.ReLU
    )
    
    linear_lr = lambda progress: learning_rate * (1 - progress)
    
    try:
        import tensorboard
        tb_log_dir = run_log_dir
        logger.info("TensorBoard logging enabled")
    except ImportError:
        tb_log_dir = None
        logger.warning("TensorBoard not installed, disabling TensorBoard logging")
    
    if load_path is not None and os.path.exists(load_path):
        logger.info(f"Loading model from {load_path}")
        model = PPO.load(
            load_path,
            env=vec_env,
            device=device,
            tensorboard_log=tb_log_dir,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=linear_lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=tb_log_dir,
            policy_kwargs=policy_kwargs,
            device=device
        )
    
    callback = TrainingCallback(verbose=1, log_dir=run_log_dir)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix=f'chair_ppo_{timestamp}'
    )
    
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=os.path.join(model_dir, 'best_model'),
        log_path=os.path.join(run_log_dir, 'eval'),
        eval_freq=max(n_steps * n_envs, 5000),
        deterministic=True,
        render=False,
        verbose=1
    )
    
    start_time = time.time()
    logger.info("Starting training...")
    
    try:
        import tqdm
        import rich
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
        logger.warning("tqdm/rich not installed, progress bar disabled")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[callback, checkpoint_callback, eval_callback],
            progress_bar=use_progress_bar
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    final_model_path = os.path.join(model_dir, f'chair_ppo_final_{timestamp}')
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    callback.plot_rewards(save_dir=run_log_dir)
    
    summary = {
        'total_episodes': len(callback.episode_rewards),
        'mean_reward': np.mean(callback.episode_rewards) if callback.episode_rewards else 0,
        'max_reward': np.max(callback.episode_rewards) if callback.episode_rewards else 0,
        'min_reward': np.min(callback.episode_rewards) if callback.episode_rewards else 0,
        'training_time_minutes': (time.time() - start_time) / 60,
        'total_timesteps': total_timesteps,
        'final_model_path': final_model_path
    }
    
    import json
    summary_path = os.path.join(run_log_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to {summary_path}")
    
    vec_env.close()
    
    return model, callback, summary


def evaluate_model(model_path, n_episodes=10, render=False):
    from env.chair_env.environment import ErgonomicChairEnv
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        env = ErgonomicChairEnv(render_mode='human' if render else None)
        obs, info = env.reset()
        
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        logger.info(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {step_count}")
        
        env.close()
    
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    logger.info(f"Max Reward: {np.max(episode_rewards):.2f}")
    logger.info(f"Min Reward: {np.min(episode_rewards):.2f}")
    logger.info("=" * 50)
    
    return episode_rewards, episode_lengths
