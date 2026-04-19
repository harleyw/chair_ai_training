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
训练监控仪表板
收集、聚合和展示分布式训练的实时指标
"""

import time
import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from training.distributed_trainer import TrainingMetrics, ClusterStatus

logger = logging.getLogger(__name__)


class AlertRule:
    """告警规则"""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[TrainingMetrics], bool],
        severity: str = "warning",
        message_template: str = ""
    ):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        
    def check(self, metrics: TrainingMetrics) -> Optional[Dict]:
        if self.condition(metrics):
            return {
                "rule": self.name,
                "severity": self.severity,
                "message": self.message_template.format(**metrics.__dict__),
                "timestamp": datetime.now().isoformat()
            }
        return None


class TrainingMonitor:
    """
    训练监控服务
    
    收集训练指标、检测异常、生成可视化数据，
    并支持通过 WebSocket 推送实时更新。
    """
    
    def __init__(self):
        self._metrics_history: List[TrainingMetrics] = []
        self._alerts: List[Dict] = []
        self._callbacks: List[Callable] = []
        
        self._lock = threading.RLock()
        self._running = False
        
        # 预定义告警规则
        self.alert_rules = [
            AlertRule(
                name="training_divergence",
                condition=lambda m: m.total_loss > 1000 and m.iteration > 100,
                severity="danger",
                message_template="训练可能发散! Loss={total_loss:.2f} 在 iteration {iteration}"
            ),
            AlertRule(
                name="low_throughput",
                condition=lambda m: m.fps < 10 and m.timestep > 5000,
                severity="warning",
                message_template="吞吐量过低: {fps:.1f} FPS"
            ),
            AlertRule(
                name="negative_reward",
                condition=lambda m: m.mean_episode_reward < -50 and m.episode > 10,
                severity="warning",
                message_template="平均奖励为负: {mean_episode_reward:.2f}"
            ),
            AlertRule(
                name="high_variance",
                condition=lambda m: m.std_episode_reward > abs(m.mean_episode_reward) * 3 and m.episode > 20,
                severity="info",
                message_template="奖励方差过大: std={std_episode_reward:.2f}"
            )
        ]
        
    @property
    def is_running(self) -> bool:
        return self._running
        
    def start(self):
        """启动监控服务"""
        with self._lock:
            self._running = True
        logger.info("TrainingMonitor started")
        
    def stop(self):
        """停止监控服务"""
        with self._lock:
            self._running = False
        logger.info("TrainingMonitor stopped")
        
    def record_metrics(self, metrics: TrainingMetrics):
        """记录新的指标数据点"""
        with self._lock:
            self._metrics_history.append(metrics)
            
            # 保留最近的数据
            if len(self._metrics_history) > 2000:
                self._metrics_history = self._metrics_history[-1000:]
                
        # 检查告警规则
        alerts = self._check_alert_rules(metrics)
        if alerts:
            with self._lock:
                self._alerts.extend(alerts)
                
        # 通知回调
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.warning(f"Monitor callback error: {e}")
                
    def _check_alert_rules(self, metrics: TrainingMetrics) -> List[Dict]:
        """检查所有告警规则"""
        triggered = []
        
        for rule in self.alert_rules:
            alert = rule.check(metrics)
            if alert:
                triggered.append(alert)
                
        return triggered
        
    def get_recent_metrics(self, n: int = 20) -> List[TrainingMetrics]:
        """获取最近的 N 条指标"""
        with self._lock:
            return list(self._metrics_history[-n:])
            
    def get_all_metrics(self) -> List[TrainingMetrics]:
        """获取所有历史指标"""
        with self._lock:
            return list(self._metrics_history)
            
    def get_alerts(self, since: Optional[datetime] = None, severity: Optional[str] = None) -> List[Dict]:
        """获取告警记录"""
        with self._lock:
            alerts = list(self._alerts)
            
        if since:
            alerts = [a for a in alerts if a.get("timestamp", "") >= since.isoformat()]
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity]
            
        return alerts
        
    def clear_alerts(self):
        """清空告警记录"""
        with self._lock:
            self._alerts.clear()
            
    def add_callback(self, callback: Callable[[TrainingMetrics], None]):
        """添加指标回调"""
        self._callbacks.append(callback)
        
    def remove_callback(self, callback: Callable):
        """移除指标回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要统计"""
        with self._lock:
            metrics = self._metrics_history
            
        if not metrics:
            return {
                "status": "no_data",
                "total_records": 0
            }
            
        latest = metrics[-1]
        
        # 计算滚动平均值
        window = min(50, len(metrics))
        recent = metrics[-window:] if len(metrics) >= window else metrics
        
        avg_loss = sum(m.total_loss for m in recent) / len(recent) if recent else 0
        avg_reward = sum(m.mean_episode_reward for m in recent) / len(recent) if recent else 0
        avg_fps = sum(m.fps for m in recent) / len(recent) if recent else 0
        
        # 趋势分析
        trend_window = min(100, len(metrics))
        older = metrics[-trend_window:-trend_window//2] if len(metrics) >= trend_window else []
        newer = metrics[-trend_window//2:] if len(metrics) >= trend_window else metrics
        
        loss_trend = "stable"
        if newer and older:
            avg_newer_loss = sum(m.total_loss for m in older) / max(len(older), 1)
            avg_newer_loss = sum(m.total_loss for m in newer) / max(len(newer), 1)
            ratio = avg_newer_loss / avg_newer_loss if avg_newer_loss > 0 else 1
            
            if ratio > 1.1:
                loss_trend = "increasing"
            elif ratio < 0.9:
                loss_trend = "decreasing"
                
        return {
            "status": "active" if self._running else "stopped",
            "total_records": len(metrics),
            "latest": {
                "iteration": latest.iteration,
                "timestep": latest.timestep,
                "episode": latest.episode,
                "loss": round(latest.total_loss, 4),
                "reward": round(latest.mean_episode_reward, 2),
                "fps": round(latest.fps, 1)
            },
            "averages": {
                "avg_loss_50episodes": round(avg_loss, 4),
                "avg_reward_50episodes": round(avg_reward, 2),
                "avg_fps": round(avg_fps, 1)
            },
            "trends": {
                "loss_trend": loss_trend
            },
            "alerts_count": len(self._alerts),
            "alert_severities": {
                "danger": len([a for a in self._alerts if a.get("severity") == "danger"]),
                "warning": len([a for a in self._alerts if a.get("severity") == "warning"]),
                "info": len([a for a in self._alerts if a.get("severity") == "info"])
            }
        }
        
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """生成前端仪表板数据"""
        summary = self.get_summary()
        recent_metrics = self.get_recent_metrics(30)
        recent_alerts = self.get_alerts(severity="danger")[:5]
        
        # Loss 曲线数据
        loss_curve = [
            {"x": m.iteration, "y": round(m.total_loss, 4)}
            for m in recent_metrics
        ]
        
        # Reward 分布数据
        rewards = [m.mean_episode_reward for m in recent_metrics]
        reward_distribution = {
            "mean": round(sum(rewards)/len(rewards), 2) if rewards else 0,
            "min": round(min(rewards), 2) if rewards else 0,
            "max": round(max(rewards), 2) if rewards else 0,
            "std": round((sum(r**2 for r in rewards)/len(rewards) - (sum(rewards)/len(rewards))**2)**0.5, 2) if rewards and len(rewards) > 1 else 0
        }
        
        # FPS 趋势
        fps_data = [
            {"x": m.iteration, "y": round(m.fps, 1)}
            for m in recent_metrics
        ]
        
        return {
            "summary": summary,
            "loss_curve": loss_curve,
            "reward_distribution": reward_distribution,
            "fps_trend": fps_data,
            "recent_alerts": recent_alerts,
            "generated_at": datetime.now().isoformat()
        }


# 全局单例
_global_monitor: Optional[TrainingMonitor] = None

def get_monitor() -> TrainingMonitor:
    """获取全局监控实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = TrainingMonitor()
    return _global_monitor
