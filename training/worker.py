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

"""
分布式训练 Worker 节点
负责本地环境实例化、数据收集、策略更新和与 Coordinator 通信
"""

import os
import time
import threading
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from training.distributed_trainer import (
    DistributedConfig,
    WorkerConfig,
    WorkerState,
    ExperienceBatch,
    GradientUpdate,
    TrainingMetrics,
)


logger = logging.getLogger(__name__)


class WorkerNode:
    """
    分布式训练 Worker 节点
    
    每个 Worker 运行一组并行环境，收集经验数据，
    并定期与 Coordinator 同步模型参数。
    
    Usage:
        worker = WorkerNode(
            worker_id="worker_0",
            config=WorkerConfig(n_envs=16),
            coordinator_url="http://coordinator:29500"
        )
        
        worker.connect()
        worker.run_collection_loop()
    """
    
    def __init__(
        self,
        worker_id: str,
        config: WorkerConfig,
        coordinator: Optional[Any] = None,
        local_mode: bool = False  # 单机模拟模式
    ):
        self.worker_id = worker_id
        self.config = config
        self.coordinator = coordinator
        self.local_mode = local_mode
        
        self._state = WorkerState.INITIALIZING
        self._vec_env = None
        self._model = None
        
        self._stats = {
            "episodes_completed": 0,
            "total_steps": 0,
            "samples_collected": 0,
            "last_heartbeat": None
        }
        
        self._stop_event = threading.Event()
        self._collection_lock = threading.RLock()
        
        self._local_buffer: List[ExperienceBatch] = []
        self._current_model_params: Optional[Dict[str, np.ndarray]] = None
        self._model_version = -1
        
        logger.info(f"WorkerNode initialized: {worker_id} (n_envs={config.n_envs})")
        
    @property
    def state(self) -> WorkerState:
        return self._state
        
    def connect(self) -> bool:
        """连接到 Coordinator"""
        try:
            if self.coordinator:
                status = self.coordinator.register_worker(self.config)
                self._state = WorkerState.READY
                logger.info(f"Connected to coordinator: {self.worker_id}")
                return True
            else:
                self._state = WorkerState.READY
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            self._state = WorkerState.ERROR
            return False
            
    def initialize_environments(self):
        """初始化本地环境池"""
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        def make_env():
            try:
                from env.chair_env.environment import ErgonomicChairEnv
                return ErgonomicChairEnv(render_mode=None)
            except Exception as e:
                logger.error(f"Failed to create environment: {e}")
                import gymnasium as gym
                return gym.make("CartPole-v1")
                
        try:
            self._vec_env = DummyVecEnv([make_env] * self.config.n_envs)
            observations = self._vec_env.reset()
            logger.info(f"Initialized {self.config.n_envs} environments (obs_shape={observations.shape})")
            return True
            
        except Exception as e:
            logger.error(f"Environment initialization failed: {e}")
            self._state = WorkerState.ERROR
            return False
            
    def run_collection_loop(
        self,
        n_steps: int = 2048,
        sync_interval: int = 100,
        max_iterations: Optional[int] = None
    ):
        """
        主数据收集循环
        
        Args:
            n_steps: 每次收集的步数
            sync_interval: 模型同步间隔 (迭代次数)
            max_iterations: 最大迭代次数 (None 表示无限)
        """
        if self._vec_env is None:
            if not self.initialize_environments():
                return
                
        self._state = WorkerState.COLLECTING
        iteration = 0
        
        logger.info(f"Starting collection loop (worker={self.worker_id})")
        
        start_time = time.time()
        
        try:
            while not self._stop_event.is_set():
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break
                    
                iteration_start = time.time()
                
                # 收集数据
                batch = self._collect_rollout(n_steps)
                
                if batch is not None and len(batch.observations) > 0:
                    # 提交到 Coordinator 或本地缓冲
                    if self.coordinator and not self.local_mode:
                        self.coordinator.submit_experience(batch)
                    else:
                        self._local_buffer.append(batch)
                        
                    # 更新统计
                    with self._collection_lock:
                        self._stats["samples_collected"] += len(batch.observations)
                        self._stats["total_steps"] += n_steps * self.config.n_envs
                        
                # 心跳和状态上报
                self._send_heartbeat()
                
                # 定期同步模型
                if iteration > 0 and iteration % sync_interval == 0:
                    self._sync_model_if_needed()
                    
                # 计算并上报指标
                elapsed = time.time() - iteration_start
                fps = (n_steps * self.config.n_envs) / elapsed if elapsed > 0 else 0
                
                metrics = TrainingMetrics(
                    iteration=iteration,
                    timestep=self._stats["total_steps"],
                    episode=self._stats["episodes_completed"],
                    policy_loss=0.0,
                    value_loss=0.0,
                    entropy_loss=0.0,
                    total_loss=0.0,
                    mean_episode_reward=0.0,
                    std_episode_reward=0.0,
                    max_episode_reward=0.0,
                    min_episode_reward=0.0,
                    mean_episode_length=0.0,
                    fps=fps,
                    collection_time_ms=elapsed * 1000,
                    aggregation_time_ms=0.0,
                    timestamp=datetime.now()
                )
                
                if self.coordinator and hasattr(self.coordinator, 'record_metrics'):
                    self.coordinator.record_metrics(metrics)
                    
                iteration += 1
                
                if iteration % 100 == 0:
                    elapsed_total = time.time() - start_time
                    logger.info(
                        f"[{self.worker_id}] Iteration {iteration}: "
                        f"steps={self._stats['total_steps']:,}, "
                        f"samples={self._stats['samples_collected']:,}, "
                        f"fps={fps:.0f}, "
                        f"elapsed={elapsed_total/60:.1f}min"
                    )
                    
        except KeyboardInterrupt:
            logger.info(f"Collection interrupted by user (worker={self.worker_id})")
            
        except Exception as e:
            logger.error(f"Collection loop error (worker={self.worker_id}): {e}", exc_info=True)
            self._state = WorkerState.ERROR
            
        finally:
            self._state = WorkerState.READY
            logger.info(f"Collection loop ended (worker={self.worker_id}), iterations={iteration}")
            
    def _collect_rollout(self, n_steps: int) -> Optional[ExperienceBatch]:
        """
        收集一个 rollout 的数据
        
        Args:
            n_steps: 收集步数
            
        Returns:
            ExperienceBatch 或 None
        """
        if self._vec_env is None:
            return None
            
        observations = self._vec_env.reset()
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_infos = []
        
        episode_rewards = []
        current_episode_reward = np.zeros(self.config.n_envs)
        
        for step in range(n_steps):
            if self._stop_event.is_set():
                break
                
            # 随机动作 (或使用策略网络)
            actions = self._vec_env.action_space.sample()
            
            next_obs, rewards, dones, truncateds, infos = self._vec_env.step(actions)
            
            real_dones = [d or t for d, t in zip(dones, truncateds)]
            
            batch_obs.append(observations)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_dones.append(real_dones)
            batch_infos.append(infos)
            
            current_episode_reward += np.array(rewards)
            
            for i, done in enumerate(real_dones):
                if done:
                    episode_rewards.append(current_episode_reward[i])
                    current_episode_reward[i] = 0
                    
            observations = next_obs
            
            # 更新 episode 统计
            with self._collection_lock:
                self._stats["episodes_completed"] += sum(real_dones)
                
        if len(batch_obs) == 0:
            return None
            
        return ExperienceBatch(
            observations=np.array(batch_obs),
            actions=np.array(batch_actions),
            rewards=np.array(batch_rewards),
            dones=np.array(batch_dones),
            infos=batch_infos,
            worker_id=self.worker_id,
            collected_at=datetime.now()
        )
        
    def _send_heartbeat(self):
        """发送心跳到 Coordinator"""
        self._stats["last_heartbeat"] = datetime.now()
        
        if self.coordinator:
            try:
                import psutil
                
                metrics = {
                    "steps": self._stats["total_steps"],
                    "episodes": self._stats["episodes_completed"],
                    "samples": self._stats["samples_collected"],
                    "cpu": psutil.cpu_percent(),
                    "memory_mb": psutil.virtual_memory().used / 1024 / 1024
                }
                
                self.coordinator.update_worker_heartbeat(self.worker_id, metrics)
                
            except Exception as e:
                logger.debug(f"Heartbeat send failed: {e}")
                
    def _sync_model_if_needed(self):
        """检查并同步模型参数（如果需要）"""
        if self.coordinator:
            try:
                remote_version = self.coordinator.get_model_version()
                
                if remote_version != self._model_version:
                    new_params = self.coordinator.get_model_params()
                    
                    if new_params:
                        self._current_model_params = new_params
                        self._model_version = remote_version
                        logger.info(
                            f"[{self.worker_id}] Model synced: version={remote_version}"
                        )
                        
            except Exception as e:
                logger.debug(f"Model sync failed: {e}")
                
    def stop(self):
        """停止 Worker"""
        self._stop_event.set()
        self._state = WorkerState.DISCONNECTED
        
        if self._vec_env:
            try:
                self._vec_env.close()
            except Exception as e:
                logger.warning(f"Error closing environments: {e}")
                
        logger.info(f"Worker stopped: {self.worker_id}")
        
    def get_local_buffer(self) -> List[ExperienceBatch]:
        """获取本地缓冲的数据"""
        with self._collection_lock:
            buffer = self._local_buffer.copy()
            self._local_buffer.clear()
            return buffer
            
    def get_stats(self) -> Dict[str, Any]:
        """获取 Worker 统计信息"""
        with self._collection_lock:
            return dict(self._stats)


class LocalDistributedSimulator:
    """
    本地分布式模拟器
    
    在单机上模拟多个 Worker 节点的行为，
    用于测试和开发阶段。
    
    Usage:
        simulator = LocalDistributedSimulator(n_workers=4)
        simulator.start()
        
        results = simulator.train(total_timesteps=50000)
        simulator.stop()
    """
    
    def __init__(
        self,
        n_workers: int = 4,
        n_envs_per_worker: int = 8,
        mode: str = "sync",
        config: Optional[DistributedConfig] = None
    ):
        self.n_workers = n_workers
        self.config = config or DistributedConfig()
        self.config.n_workers = n_workers
        self.config.n_envs_per_worker = n_envs_per_worker
        self.config.mode = mode
        
        self.trainer = None
        self.workers: List[WorkerNode] = []
        self._running = False
        
    def start(self) -> Dict[str, Any]:
        """启动模拟的分布式训练"""
        from training.distributed_trainer import DistributedTrainer
        
        self.trainer = DistributedTrainer(config=self.config)
        cluster_status = self.trainer.start()
        
        # 创建本地 Workers
        self.workers = []
        for i in range(self.n_workers):
            worker_config = WorkerConfig(
                worker_id=f"worker_{i}",
                n_envs=self.config.n_envs_per_worker
            )
            
            worker = WorkerNode(
                worker_id=worker_config.worker_id,
                config=worker_config,
                coordinator=self.trainer,
                local_mode=True
            )
            
            worker.connect()
            worker.initialize_environments()
            self.workers.append(worker)
            
        self._running = True
        logger.info(f"LocalDistributedSimulator started with {n_workers} workers")
        
        return {
            "status": "running",
            "n_workers": n_workers,
            "cluster_status": cluster_status,
            "workers": [w.worker_id for w in self.workers]
        }
        
    def train(
        self,
        total_timesteps: int = 100000,
        n_steps: int = 2048,
        sync_interval: int = 50
    ) -> Dict[str, Any]:
        """
        执行分布式训练
        
        Args:
            total_timesteps: 总训练步数
            n_steps: 每次 rollout 步数
            sync_interval: 同步间隔
            
        Returns:
            训练结果
        """
        if not self._running or not self.workers:
            raise RuntimeError("Simulator not started")
            
        start_time = time.time()
        
        threads = []
        for worker in self.workers:
            t = threading.Thread(
                target=worker.run_collection_loop,
                kwargs={
                    "n_steps": n_steps,
                    "sync_interval": sync_interval,
                    "max_iterations": total_timesteps // (n_steps * self.config.n_envs_per_worker)
                }
            )
            t.daemon = True
            t.start()
            threads.append(t)
            
        # 主线程监控
        while any(t.is_alive() for t in threads) and self._running:
            time.sleep(1)
            
            # 上报状态
            if self.trainer and self.trainer.is_running:
                status = self.trainer.get_cluster_status()
                if status.active_workers > 0:
                    logger.info(
                        f"[Simulator] timesteps={status.total_timesteps:,}, "
                        f"active_workers={status.active_workers}, "
                        f"fps={status.throughput_fps:.0f}"
                    )
                    
        # 等待所有线程结束
        for t in threads:
            t.join(timeout=10)
            
        training_time = time.time() - start_time
        
        # 收集结果
        all_stats = [w.get_stats() for w in self.workers]
        total_samples = sum(s.get("samples_collected", 0) for s in all_stats)
        total_episodes = sum(s.get("episodes_completed", 0) for s in all_stats)
        
        result = {
            "success": True,
            "training_time_seconds": round(training_time, 2),
            "training_time_minutes": round(training_time / 60, 2),
            "total_samples": total_samples,
            "total_episodes": total_episodes,
            "throughput_fps": round(total_samples / training_time, 1) if training_time > 0 else 0,
            "n_workers": self.n_workers,
            "n_envs_total": self.n_workers * self.config.n_envs_per_worker,
            "worker_stats": all_stats,
            "cluster_status": self.trainer.get_cluster_status() if self.trainer else None
        }
        
        logger.info(f"Distributed training completed: {result}")
        return result
        
    def stop(self) -> Dict[str, Any]:
        """停止所有 Worker 和协调器"""
        self._running = False
        
        for worker in self.workers:
            worker.stop()
            
        if self.trainer:
            final_status = self.trainer.stop()
        else:
            final_status = {"status": "stopped"}
            
        self.workers.clear()
        
        logger.info("LocalDistributedSimulator stopped")
        return final_status
        
    def get_worker_stats(self) -> List[Dict[str, Any]]:
        """获取所有 Worker 的统计信息"""
        return [w.get_stats() for w in self.workers]
