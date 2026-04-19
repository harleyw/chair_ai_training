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

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np


class SensorData(BaseModel):
    """传感器数据输入模型"""
    pressure_matrix: List[List[float]] = Field(
        ...,
        description="8x8 压力传感器阵列数据 (0-1 归一化)",
        example=[[0.1]*8 for _ in range(8)]
    )
    posture_angles: List[float] = Field(
        ...,
        description="姿态角度 [头部角度, 肩部角度, 骨盆角度] (度)",
        example=[15.0, -5.0, 10.0]
    )
    sitting_duration: float = Field(
        ...,
        ge=0,
        description="静坐时长 (秒)",
        example=1800.0
    )
    user_weight: float = Field(
        ...,
        ge=30,
        le=200,
        description="用户体重 (kg)",
        example=70.0
    )
    user_height: float = Field(
        ...,
        ge=1.0,
        le=2.5,
        description="用户身高 (m)",
        example=1.70
    )
    fatigue_level: float = Field(
        0.0,
        ge=0,
        le=1,
        description="疲劳程度 (0-1)",
        example=0.3
    )


class ChairAction(BaseModel):
    """座椅调整动作输出模型"""
    seat_height: float = Field(
        ...,
        ge=-1,
        le=1,
        description="座垫高度调整 (-1 到 1)"
    )
    backrest_angle: float = Field(
        ...,
        ge=-1,
        le=1,
        description="靠背角度调整 (-1 到 1)"
    )
    lumbar_position: float = Field(
        ...,
        ge=-1,
        le=1,
        description="腰托位置调整 (-1 到 1)"
    )
    lumbar_thickness: float = Field(
        ...,
        ge=-1,
        le=1,
        description="腰托厚度调整 (-1 到 1)"
    )
    headrest_position: float = Field(
        ...,
        ge=-1,
        le=1,
        description="头枕位置调整 (-1 到 1)"
    )
    headrest_angle: float = Field(
        ...,
        ge=-1,
        le=1,
        description="头枕角度调整 (-1 到 1)"
    )
    left_armrest: float = Field(
        ...,
        ge=-1,
        le=1,
        description="左扶手高度调整 (-1 到 1)"
    )
    right_armrest: float = Field(
        ...,
        ge=-1,
        le=1,
        description="右扶手高度调整 (-1 到 1)"
    )


class DetailedPostureAnalysis(BaseModel):
    """详细姿态分析结果模型（v2.0 新增）"""
    posture_type: str = Field(
        ...,
        description="具体姿态类型: normal/forward_lean/backward_recline/lateral_tilt/crossed_legs/leg_crossed/lotus_position/forward_reach"
    )
    posture_name_cn: str = Field(
        ...,
        description="姿态中文名称"
    )
    severity: str = Field(
        ...,
        description="严重程度: ideal/good/warning/danger"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="分类置信度 (0-1)"
    )
    risk_areas: List[str] = Field(
        default_factory=list,
        description="受影响的身体部位"
    )
    recommended_exercises: List[str] = Field(
        default_factory=list,
        description="推荐的矫正动作/练习"
    )
    primary_adjustments: Dict[str, float] = Field(
        default_factory=dict,
        description="主要调整建议 {参数名: 调整值}"
    )
    message: str = Field(
        default="",
        description="人性化调整建议消息"
    )


class AdjustmentRecommendation(BaseModel):
    """调整建议响应模型"""
    success: bool = Field(..., description="请求是否成功")
    timestamp: str = Field(..., description="响应时间戳")
    action: ChairAction = Field(..., description="推荐的座椅调整动作")
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="置信度 (0-1)"
    )
    comfort_score: float = Field(
        ...,
        description="预测舒适度评分"
    )
    pressure_risk: str = Field(
        ...,
        description="压力风险等级: low/medium/high"
    )
    posture_analysis: dict = Field(
        ...,
        description="姿态分析结果"
    )
    recommendations: List[str] = Field(
        ...,
        description="人性化调整建议"
    )
    posture_detail: Optional[DetailedPostureAnalysis] = Field(
        None,
        description="详细姿态分类结果（v2.0 新增，向后兼容）"
    )


class BatchSensorData(BaseModel):
    """批量传感器数据输入"""
    samples: List[SensorData] = Field(
        ...,
        description="多个时间点的传感器数据样本",
        min_length=1,
        max_length=100
    )


class BatchAdjustmentResponse(BaseModel):
    """批量调整响应"""
    success: bool
    total_samples: int
    results: List[AdjustmentRecommendation]
    processing_time_ms: float


class ModelInfo(BaseModel):
    """模型信息响应"""
    model_name: str
    version: str
    status: str
    loaded_at: Optional[str] = None
    training_info: Optional[dict] = None
    observation_space: int
    action_space: int
    algorithm: str


class HealthStatus(BaseModel):
    """健康检查响应"""
    status: str
    service: str
    version: str
    model_loaded: bool
    uptime_seconds: float
    timestamp: str


class TrainingConfig(BaseModel):
    """训练配置输入"""
    total_timesteps: int = Field(100000, ge=1000)
    learning_rate: float = Field(3e-4, gt=0, lt=1)
    n_steps: int = Field(2048, ge=128)
    batch_size: int = Field(64, ge=16)
    save_dir: str = Field("./models", description="模型保存目录")


class TrainingResponse(BaseModel):
    """训练响应"""
    success: bool
    message: str
    training_id: Optional[str] = None
    config: TrainingConfig
    estimated_time_minutes: Optional[float] = None


class ExportResponse(BaseModel):
    """模型导出响应"""
    success: bool
    message: str
    format: str = "onnx"
    output_path: Optional[str] = None
    file_size_mb: Optional[float] = None
    export_info: Optional[Dict[str, Any]] = None


class ExportedList(BaseModel):
    """已导出模型列表响应"""
    success: bool
    total_models: int
    models: List[Dict[str, Any]]
