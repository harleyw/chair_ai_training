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
分布式训练 REST API 路由
提供训练任务管理、Worker 管理、监控查询等端点
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from training.distributed_trainer import (
    DistributedTrainer,
    DistributedConfig,
    TrainingMode,
    AggregationMethod,
    WorkerConfig,
    WorkerState,
    ClusterStatus,
    TrainingMetrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/training/distributed",
    tags=["分布式训练"]
)

# 全局存储: job_id -> trainer 实例
_active_trainers: Dict[str, DistributedTrainer] = {}


class StartTrainingRequest(BaseModel):
    """启动分布式训练请求"""
    n_workers: int = Field(4, ge=1, le=64, description="Worker 节点数量")
    n_envs_per_worker: int = Field(16, ge=1, le=128, description="每个 Worker 的环境数")
    
    # 训练参数
    total_timesteps: int = Field(1000000, ge=1000, description="总训练步数")
    learning_rate: float = Field(3e-4, gt=0, le=1.0, description="学习率")
    batch_size: int = Field(64, ge=32, le=512, description="批次大小")
    n_epochs: int = Field(10, ge=1, le=50, description="PPO epoch 数")
    
    # 分布式配置
    mode: str = Field("sync", pattern="^(sync|async|ssp)$")
    aggregation_method: str = Field("mean", pattern="^(mean|weighted|fedavg)$")
    sync_interval: int = Field(100, ge=10, le=10000, description="同步间隔 (步)")
    
    # 可选配置
    gpu_per_worker: Optional[int] = Field(None, ge=0, le=8)
    auto_restart: bool = Field(True, description="Worker 失败时自动重启")
    checkpoint_interval: int = Field(10000, ge=1000)


class ScaleRequest(BaseModel):
    """扩缩容请求"""
    target_workers: int = Field(..., ge=1, le=128)


class TrainingJobInfo(BaseModel):
    """训练任务信息"""
    job_id: str
    status: str
    config: DistributedConfig
    cluster_status: ClusterStatus
    created_at: datetime
    updated_at: datetime


# ==================== 训练生命周期管理 ====================

@router.post("/start", response_model=TrainingJobInfo)
async def start_distributed_training(request: StartTrainingRequest):
    """
    启动分布式训练任务
    
    创建新的协调器实例，配置 Worker 池，开始等待 Workers 连接。
    
    - **request**: 训练配置请求体
    
    Returns:
        创建的训练任务信息，包含 job_id 和初始状态
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    config = DistributedConfig(
        mode=TrainingMode(request.mode),
        n_workers=request.n_workers,
        n_envs_per_worker=request.n_envs_per_worker,
        
        aggregation_method=AggregationMethod(request.aggregation_method),
        sync_interval=request.sync_interval,
        
        learning_rate=request.learning_rate,
        n_steps=2048,
        batch_size=request.batch_size,
        n_epochs=request.n_epochs,
        
        auto_restart_workers=request.auto_restart,
        checkpoint_interval=request.checkpoint_interval
    )
    
    trainer = DistributedTrainer(config=config)
    _active_trainers[job_id] = trainer
    
    status = trainer.start()
    
    logger.info(f"Distributed training started: {job_id} (workers={request.n_workers})")
    
    return TrainingJobInfo(
        job_id=job_id,
        status=status.status,
        config=config,
        cluster_status=status,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


@router.get("/status/{job_id}", response_model=ClusterStatus)
async def get_training_status(job_id: str):
    """
    查询训练任务状态
    
    获取指定训练任务的详细状态，包括 Worker 健康情况、训练进度、性能指标。
    
    - **job_id**: 任务 ID
    
    Returns:
        集群完整状态
    """
    trainer = _active_trainers.get(job_id)
    if not trainer:
        raise HTTPException(status_code=404, detail=f"任务 '{job_id}' 不存在或已结束")
    
    return trainer.get_cluster_status()


@router.post("/pause/{job_id}", response_model=ClusterStatus)
async def pause_training(job_id: str):
    """暂停指定训练任务"""
    trainer = _active_trainers.get(job_id)
    if not trainer:
        raise HTTPException(status_code=404, detail=f"任务 '{job_id}' 不存在")
    
    return trainer.pause()


@router.post("/resume/{job_id}", response_model=ClusterStatus)
async def resume_training(job_id: str):
    """恢复已暂停的训练任务"""
    trainer = _active_trainers.get(job_id)
    if not trainer:
        raise HTTPException(status_code=404, detail=f"任务 '{job_id}' 不存在")
    
    return trainer.resume()


@router.post("/stop/{job_id}", response_model=ClusterStatus)
async def stop_training(job_id: str):
    """停止训练任务并释放资源"""
    trainer = _active_trainers.pop(job_id, None)
    if not trainer:
        raise HTTPException(status_code=404, detail=f"任务 '{job_id}' 不存在或已结束")
    
    status = trainer.stop()
    logger.info(f"Training stopped: {job_id}")
    
    return status


# ==================== Worker 管理 ====================

@router.get("/workers", response_model=List[Dict])
async def list_all_workers():
    """列出所有活跃任务的 Worker 状态"""
    all_workers = []
    
    for job_id, trainer in _active_trainers.items():
        workers = trainer.get_worker_statuses()
        for wid, wstatus in workers.items():
            all_workers.append({
                "job_id": job_id,
                "worker_id": wid,
                "state": wstatus.state.value,
                "n_envs": wstatus.config.n_envs if wstatus.config else 0,
                "total_steps": wstatus.total_steps,
                "episodes": wstatus.episodes_completed,
                "samples": wstatus.samples_collected,
                "last_heartbeat": wstatus.last_heartbeat.isoformat() if wstatus.last_heartbeat else None,
                "error": wstatus.error_message
            })
            
    return all_workers


@router.get("/workers/health", response_model=Dict[str, str])
async def check_workers_health():
    """检查所有 Worker 健康状态"""
    health_results = {}
    
    for job_id, trainer in _active_trainers.items():
        worker_health = trainer.check_worker_health()
        for wid, health in worker_health.items():
            key = f"{job_id}:{wid}"
            health_results[key] = health
            
    return health_results or {"message": "No active workers"}


@router.post("/scale", response_model=Dict[str, Any])
async def scale_workers(request: ScaleRequest):
    """
    弹性扩缩容 Worker 数量
    
    返回扩缩容建议和当前集群状态。
    注意：实际扩缩容需要配合容器编排系统（K8s）使用。
    """
    # 找到第一个活跃的 trainer
    trainer = next(iter(_active_trainers.values()), None)
    if not trainer:
        raise HTTPException(status_code=400, detail="没有活跃的训练任务")
    
    result = trainer.scale_workers(request.target_workers)
    
    result["current_jobs"] = list(_active_trainers.keys())
    result["timestamp"] = datetime.now().isoformat()
    
    return result


# ==================== 监控指标 ====================

@router.get("/metrics/{job_id}")
async def get_training_metrics(
    job_id: str,
    last_n: int = Query(20, ge=1, le=100)
):
    """
    获取训练指标历史
    
    返回最近的训练指标数据点，包括 loss、reward、FPS 等。
    
    - **job_id**: 任务 ID
    - **last_n**: 返回最近 N 条记录
    """
    trainer = _active_trainers.get(job_id)
    if not trainer:
        raise HTTPException(status_code=404, detail=f"任务 '{job_id}' 不存在")
    
    metrics = trainer.get_training_metrics(last_n=last_n)
    
    return {
        "job_id": job_id,
        "count": len(metrics),
        "metrics": [
            {
                "iteration": m.iteration,
                "timestep": m.timestep,
                "episode": m.episode,
                "policy_loss": round(m.policy_loss, 6),
                "value_loss": round(m.value_loss, 6),
                "entropy_loss": round(m.entropy_loss, 6),
                "total_loss": round(m.total_loss, 6),
                "mean_reward": round(m.mean_episode_reward, 2),
                "std_reward": round(m.std_episode_reward, 2),
                "max_reward": round(m.max_episode_reward, 2),
                "min_reward": round(m.min_episode_reward, 2),
                "fps": round(m.fps, 1),
                "collection_time_ms": round(m.collection_time_ms, 2),
                "timestamp": m.timestamp.isoformat()
            }
            for m in metrics
        ]
    }


@router.get("/cluster/status")
async def get_global_cluster_status():
    """获取全局集群状态概览"""
    jobs_status = []
    
    for job_id, trainer in _active_trainers.items():
        status = trainer.get_cluster_status()
        workers = trainer.get_worker_statuses()
        
        jobs_status.append({
            "job_id": job_id,
            "status": status.status,
            "total_workers": status.total_workers,
            "active_workers": status.active_workers,
            "error_workers": status.error_workers,
            "timesteps": status.total_timesteps,
            "episodes": status.total_episodes,
            "fps": round(status.throughput_fps, 1),
            "elapsed_seconds": round(status.elapsed_seconds, 1),
            "worker_details": [
                {
                    "id": wid,
                    "state": ws.state.value,
                    "steps": ws.total_steps
                }
                for wid, ws in workers.items()
            ]
        })
    
    return {
        "active_jobs": len(_active_trainers),
        "total_workers": sum(s["total_workers"] for s in jobs_status),
        "active_workers": sum(s["active_workers"] for s in jobs_status),
        "jobs": jobs_status,
        "timestamp": datetime.now().isoformat()
    }


# ==================== 快速启动本地模拟 ====================

class LocalSimulatorRequest(BaseModel):
    """本地模拟训练请求"""
    n_workers: int = Field(4, ge=1, le=16, description="模拟 Worker 数量")
    n_envs_per_worker: int = Field(8, ge=1, le=32, description="每个 Worker 环境数")
    total_timesteps: int = Field(50000, ge=1000, description="总训练步数")
    mode: str = Field("sync", pattern="^(sync|async|ssp)$")


class SimulatorResult(BaseModel):
    """模拟结果"""
    success: bool
    training_time_seconds: float
    training_time_minutes: float
    total_samples: int
    total_episodes: int
    throughput_fps: float
    n_workers: int
    n_envs_total: int
    worker_stats: List[Dict]


@router.post("/simulate/local", response_model=SimulatorResult)
async def run_local_simulation(request: LocalSimulatorRequest):
    """
    本地模拟分布式训练
    
    在单机上模拟多个 Worker 并行训练的行为，
    用于快速测试和开发。无需真实的分布式基础设施。
    
    - **request**: 模拟配置
    
    Returns:
        模拟训练结果，包含性能统计
    """
    from training.worker import LocalDistributedSimulator
    
    simulator = LocalDistributedSimulator(
        n_workers=request.n_workers,
        n_envs_per_worker=request.n_envs_per_worker,
        mode=request.mode
    )
    
    try:
        simulator.start()
        
        result = simulator.train(
            total_timesteps=request.total_timesteps,
            sync_interval=max(10, request.total_timesteps // 1000)
        )
        
        final_status = simulator.stop()
        
        return SimulatorResult(**result)
        
    except Exception as e:
        logger.error(f"Local simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 预设配置模板 ====================

@router.get("/templates", response_model=List[Dict])
async def list_training_templates():
    """列出预定义的训练配置模板"""
    templates = [
        {
            "name": "quick_test",
            "display_name": "快速测试",
            "description": "小规模快速验证，适合开发调试",
            "config": {
                "n_workers": 2,
                "n_envs_per_worker": 4,
                "total_timesteps": 10000,
                "mode": "sync"
            }
        },
        {
            "name": "standard",
            "display_name": "标准训练",
            "description": "平衡性能和资源消耗的标准配置",
            "config": {
                "n_workers": 4,
                "n_envs_per_worker": 16,
                "total_timesteps": 500000,
                "mode": "sync"
            }
        },
        {
            "name": "large_scale",
            "display_name": "大规模训练",
            "description": "多节点高性能训练，适合最终模型训练",
            "config": {
                "n_workers": 8,
                "n_envs_per_worker": 32,
                "total_timesteps": 2000000,
                "mode": "async"
            }
        },
        {
            "name": "research_experiment",
            "display_name": "研究实验",
            "description": "用于超参数搜索和 A/B 测试",
            "config": {
                "n_workers": 16,
                "n_envs_per_worker": 8,
                "total_timesteps": 200000,
                "mode": "ssp"
            }
        }
    ]
    
    return templates
