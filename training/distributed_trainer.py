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
分布式训练协调器
管理 Worker 节点池、梯度聚合、模型同步和任务调度
"""

import time
import threading
import uuid
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """分布式训练模式"""
    SYNC = "sync"          # 同步：所有 Worker 完成后聚合
    ASYNC = "async"        # 异步：Worker 独立更新，定期同步
    SSP = "ssp"            # Stale Synchronous Parallel


class WorkerState(Enum):
    """Worker 节点状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    COLLECTING = "collecting"
    SYNCING = "syncing"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class AggregationMethod(Enum):
    """梯度聚合方法"""
    MEAN = "mean"          # 平均
    WEIGHTED = "weighted"   # 加权平均 (按数据量)
    FEDAVG = "fedavg"       # 联邦平均


@dataclass
class WorkerConfig:
    """Worker 配置"""
    worker_id: str
    n_envs: int = 16
    gpu_id: Optional[int] = None
    batch_size: int = 64
    n_steps: int = 2048


@dataclass
class WorkerStatus:
    """Worker 状态信息"""
    worker_id: str
    state: WorkerState = WorkerState.INITIALIZING
    config: Optional[WorkerConfig] = None
    
    # 性能指标
    episodes_completed: int = 0
    total_steps: int = 0
    samples_collected: int = 0
    last_heartbeat: Optional[datetime] = None
    
    # 资源使用
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # 错误信息
    error_message: Optional[str] = None
    error_count: int = 0
    
    connected_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class DistributedConfig:
    """分布式训练配置"""
    mode: TrainingMode = TrainingMode.SYNC
    n_workers: int = 4
    n_envs_per_worker: int = 16
    
    # 聚合配置
    aggregation_method: AggregationMethod = AggregationMethod.MEAN
    sync_interval: int = 100  # 每 N 步同步一次
    
    # 超时配置
    heartbeat_timeout_seconds: float = 30.0
    worker_timeout_seconds: float = 120.0
    max_retries: int = 3
    
    # 训练参数 (传递给 PPO)
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    
    # 模型配置
    policy_net_arch: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # 容错配置
    auto_restart_workers: bool = True
    checkpoint_interval: int = 10000
    
    # 监控配置
    metrics_interval_seconds: float = 10.0
    enable_dashboard: bool = True


@dataclass
class ExperienceBatch:
    """经验批次"""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict]
    worker_id: str
    collected_at: datetime = field(default_factory=datetime.now)


@dataclass
class GradientUpdate:
    """梯度更新"""
    gradients: Dict[str, np.ndarray]
    worker_id: str
    step_count: int
    loss_value: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ClusterStatus:
    """集群状态"""
    job_id: str
    status: str  # running | paused | stopped | completed | error
    coordinator_host: str
    started_at: Optional[datetime]
    
    # Worker 统计
    total_workers: int = 0
    active_workers: int = 0
    ready_workers: int = 0
    error_workers: int = 0
    
    # 训练统计
    total_timesteps: int = 0
    total_episodes: int = 0
    current_iteration: int = 0
    
    # 性能统计
    avg_reward: float = 0.0
    avg_loss: float = 0.0
    throughput_fps: float = 0.0
    
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None


@dataclass
class TrainingMetrics:
    """训练指标"""
    iteration: int
    timestep: int
    episode: int
    
    # Loss 分量
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    
    # 奖励指标
    mean_episode_reward: float
    std_episode_reward: float
    max_episode_reward: float
    min_episode_reward: float
    mean_episode_length: float
    
    # 性能指标
    fps: float
    collection_time_ms: float
    aggregation_time_ms: float
    
    # Worker 维度
    worker_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.now)


class ExperienceBuffer:
    """共享经验缓冲区"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer: List[ExperienceBatch] = []
        self._lock = threading.RLock()
        self.total_samples = 0
        
    def add(self, batch: ExperienceBatch):
        with self._lock:
            self.buffer.append(batch)
            self.total_samples += len(batch.observations)
            if len(self.buffer) > self.max_size * 2:
                self.buffer = self.buffer[-self.max_size:]
                
    def get_all(self) -> List[ExperienceBatch]:
        with self._lock:
            batches = self.buffer.copy()
            self.buffer.clear()
            return batches
            
    def size(self) -> int:
        with self._lock:
            return sum(len(b.observations) for b in self.buffer)
            
    def clear(self):
        with self._lock:
            self.buffer.clear()


class DistributedTrainer:
    """
    分布式训练协调器
    
    管理 Worker 节点池，协调数据收集、梯度聚合和模型同步。
    
    Usage:
        trainer = DistributedTrainer(config)
        trainer.start()
        
        # Workers 连接...
        
        result = trainer.train(total_timesteps=1000000)
        trainer.stop()
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.job_id = str(uuid.uuid4())[:8]
        
        self.workers: Dict[str, WorkerStatus] = {}
        self.experience_buffer = ExperienceBuffer()
        
        self._state = "initialized"
        self._start_time: Optional[datetime] = None
        self._iteration = 0
        self._total_timesteps = 0
        self._total_episodes = 0
        
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        self._metrics_history: List[TrainingMetrics] = []
        self._callbacks: List[Callable] = []
        
        self._model_params: Optional[Dict[str, np.ndarray]] = None
        self._model_version = 0
        
        logger.info(f"DistributedTrainer initialized with job_id={self.job_id}")
        
    @property
    def state(self) -> str:
        return self._state
        
    @property
    def is_running(self) -> bool:
        return self._state == "running"
        
    @property
    def active_worker_ids(self) -> List[str]:
        return [
            wid for wid, wstatus in self.workers.items()
            if wstatus.state in [WorkerState.READY, WorkerState.COLLECTING]
        ]
        
    def start(self) -> ClusterStatus:
        """启动协调器服务"""
        with self._lock:
            self._state = "running"
            self._start_time = datetime.now()
            
            logger.info(f"Coordinator starting (job_id={self.job_id})")
            logger.info(f"Mode: {self.config.mode.value}")
            logger.info(f"Expected workers: {self.config.n_workers}")
            
        return self.get_cluster_status()
        
    def stop(self) -> ClusterStatus:
        """停止所有训练并清理资源"""
        with self._lock:
            self._state = "stopped"
            self._stop_event.set()
            
            for worker_id in list(self.workers.keys()):
                self.workers[worker_id].state = WorkerState.DISCONNECTED
                
            logger.info(f"Coordinator stopped (job_id={self.job_id})")
            
        return self.get_cluster_status()
        
    def pause(self) -> ClusterStatus:
        """暂停训练"""
        with self._lock:
            if self._state == "running":
                self._state = "paused"
                logger.info(f"Training paused (job_id={self.job_id})")
        return self.get_cluster_status()
        
    def resume(self) -> ClusterStatus:
        """恢复训练"""
        with self._lock:
            if self._state == "paused":
                self._state = "running"
                logger.info(f"Training resumed (job_id={self.job_id})")
        return self.get_cluster_status()

    def register_worker(self, worker_config: WorkerConfig) -> WorkerStatus:
        """
        注册新 Worker
        
        Args:
            worker_config: Worker 配置
            
        Returns:
            注册后的 Worker 状态
        """
        with self._lock:
            worker_id = worker_config.worker_id
            
            if worker_id in self.workers:
                existing = self.workers[worker_id]
                existing.config = worker_config
                existing.state = WorkerState.READY
                existing.connected_at = datetime.now()
                existing.updated_at = datetime.now()
                existing.error_message = None
                logger.info(f"Worker re-registered: {worker_id}")
                return existing
            
            status = WorkerStatus(
                worker_id=worker_id,
                state=WorkerState.READY,
                config=worker_config,
                connected_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.workers[worker_id] = status
            
            logger.info(
                f"Worker registered: {worker_id} "
                f"(n_envs={worker_config.n_envs}, total_workers={len(self.workers)})"
            )
            
            return status
            
    def unregister_worker(self, worker_id: str) -> bool:
        """移除 Worker"""
        with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Worker unregistered: {worker_id}")
                return True
            return False
            
    def update_worker_heartbeat(self, worker_id: str, metrics: Optional[Dict] = None) -> bool:
        """
        更新 Worker 心跳
        
        Args:
            worker_id: Worker ID
            metrics: 可选的性能指标
            
        Returns:
            是否成功更新
        """
        with self._lock:
            if worker_id not in self.workers:
                return False
                
            status = self.workers[worker_id]
            status.last_heartbeat = datetime.now()
            status.updated_at = datetime.now()
            
            if metrics:
                status.cpu_usage = metrics.get("cpu", status.cpu_usage)
                status.memory_usage_mb = metrics.get("memory_mb", status.memory_usage_mb)
                status.gpu_usage = metrics.get("gpu", status.gpu_usage)
                status.gpu_memory_mb = metrics.get("gpu_memory_mb", status.gpu_memory_mb)
                status.total_steps = metrics.get("steps", status.total_steps)
                status.episodes_completed = metrics.get("episodes", status.episodes_completed)
                status.samples_collected = metrics.get("samples", status.samples_collected)
                
            if status.state not in [WorkerState.ERROR, WorkerState.DISCONNECTED]:
                status.state = WorkerState.COLLECTING
                
            return True
            
    def submit_experience(self, batch: ExperienceBatch) -> bool:
        """
        Worker 提交经验数据
        
        Args:
            batch: 经验批次
            
        Returns:
            是否接受
        """
        if self._state != "running":
            return False
            
        self.experience_buffer.add(batch)
        
        with self._lock:
            if batch.worker_id in self.workers:
                self.workers[batch.worker_id].samples_collected += len(batch.observations)
                
        return True
        
    def submit_gradient(self, gradient: GradientUpdate) -> Optional[Dict]:
        """
        Worker 提交梯度更新
        
        Args:
            gradient: 梯度数据
            
        Returns:
            聚合后的全局模型参数 (如果是同步模式)，或 None (异步模式)
        """
        if self._state != "running":
            return None
            
        with self._lock:
            self._iteration += 1
            self._total_timesteps += gradient.step_count
            
            if gradient.worker_id in self.workers:
                self.workers[gradient.worker_id].total_steps += gradient.step_count
                
        if self.config.mode == TrainingMode.SYNC:
            aggregated = self._aggregate_gradients([gradient])
            if aggregated:
                self._model_params = aggregated
                self._model_version += 1
            return aggregated
            
        elif self.config.mode == TrainingMode.ASYNC:
            if self._iteration % self.config.sync_interval == 0:
                pending = self._get_pending_gradients()
                if pending:
                    aggregated = self._aggregate_gradients(pending)
                    if aggregated:
                        self._model_params = aggregated
                        self._model_version += 1
                    return aggregated
            return None
            
        return None
        
    def _get_pending_gradients(self) -> List[GradientUpdate]:
        """获取待处理的梯度 (简化实现)"""
        return []
        
    def _aggregate_gradients(self, gradients: List[GradientUpdate]) -> Optional[Dict[str, np.ndarray]]:
        """
        聚合多个 Worker 的梯度
        
        Args:
            gradients: 梯度列表
            
        Returns:
            聚合后的参数字典
        """
        if not gradients:
            return None
            
        method = self.config.aggregation_method
        
        if method == AggregationMethod.MEAN:
            return self._mean_aggregation(gradients)
        elif method == AggregationMethod.WEIGHTED:
            return self._weighted_aggregation(gradients)
        elif method == AggregationMethod.FEDAVG:
            return self._fedavg_aggregation(gradients)
        else:
            return self._mean_aggregation(gradients)
            
    def _mean_aggregation(self, gradients: List[GradientUpdate]) -> Dict[str, np.ndarray]:
        """平均聚合"""
        if not gradients or not gradients[0].gradients:
            return {}
            
        result = {}
        for key in gradients[0].gradients.keys():
            stacked = np.stack([g.gradients[key] for g in gradients])
            result[key] = np.mean(stacked, axis=0)
        return result
        
    def _weighted_aggregation(self, gradients: List[GradientUpdate]) -> Dict[str, np.ndarray]:
        """加权聚合 (按样本数)"""
        weights = np.array([g.step_count for g in gradients], dtype=np.float32)
        weights = weights / weights.sum()
        
        result = {}
        for key in gradients[0].gradients.keys():
            weighted_sum = sum(w * g.gradients[key] for w, g in zip(weights, gradients))
            result[key] = weighted_sum
        return result
        
    def _fedavg_aggregation(self, gradients: List[GradientUpdate]) -> Dict[str, np.ndarray]:
        """联邦平均 (FedAvg)"""
        return self._mean_aggregation(gradients)
        
    def get_model_params(self) -> Optional[Dict[str, np.ndarray]]:
        """获取当前模型参数"""
        return self._model_params
        
    def get_model_version(self) -> int:
        """获取当前模型版本号"""
        return self._model_version
        
    def get_cluster_status(self) -> ClusterStatus:
        """获取集群完整状态"""
        active = sum(1 for w in self.workers.values() 
                     if w.state in [WorkerState.READY, WorkerState.COLLECTING])
        ready = sum(1 for w in self.workers.values() 
                   if w.state == WorkerState.READY)
        errors = sum(1 for w in self.workers.values() 
                   if w.state == WorkerState.ERROR)
                   
        elapsed = 0.0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            
        avg_fps = 0.0
        if self.workers and elapsed > 0:
            total_steps = sum(w.total_steps for w in self.workers.values())
            avg_fps = total_steps / elapsed
            
        return ClusterStatus(
            job_id=self.job_id,
            status=self._state,
            coordinator_host="localhost",
            started_at=self._start_time,
            total_workers=len(self.workers),
            active_workers=active,
            ready_workers=ready,
            error_workers=errors,
            total_timesteps=self._total_timesteps,
            total_episodes=self._total_episodes,
            current_iteration=self._iteration,
            throughput_fps=avg_fps,
            elapsed_seconds=elapsed
        )
        
    def get_worker_statuses(self) -> Dict[str, WorkerStatus]:
        """获取所有 Worker 状态"""
        return dict(self.workers)
        
    def get_training_metrics(self, last_n: int = 10) -> List[TrainingMetrics]:
        """获取最近的训练指标"""
        return self._metrics_history[-last_n:] if self._metrics_history else []
        
    def record_metrics(self, metrics: TrainingMetrics):
        """记录训练指标"""
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-500:]
            
    def add_callback(self, callback: Callable[[TrainingMetrics], None]):
        """添加训练回调"""
        self._callbacks.append(callback)
        
    def check_worker_health(self) -> Dict[str, str]:
        """
        检查所有 Worker 健康状态
        
        Returns:
            {worker_id: "healthy"|"stale"|"error"| "missing"}
        """
        results = {}
        now = datetime.now()
        
        for worker_id, status in self.workers.items():
            if status.last_heartbeat is None:
                results[worker_id] = "missing"
            else:
                delta = (now - status.last_heartbeat).total_seconds()
                if delta > self.config.worker_timeout_seconds:
                    results[worker_id] = "stale"
                    status.state = WorkerState.DISCONNECTED
                elif status.state == WorkerState.ERROR:
                    results[worker_id] = "error"
                else:
                    results[worker_id] = "healthy"
                    
        return results
        
    def scale_workers(self, target_count: int) -> Dict[str, any]:
        """
        弹性扩缩容 Worker 数量
        
        Args:
            target_count: 目标 Worker 数量
            
        Returns:
            操作结果
        """
        current = len(self.workers)
        action = "scale_up" if target_count > current else "scale_down"
        
        return {
            "action": action,
            "current_workers": current,
            "target_workers": target_count,
            "delta": target_count - current,
            "message": f"Scaling from {current} to {target_count} workers ({action})"
        }
