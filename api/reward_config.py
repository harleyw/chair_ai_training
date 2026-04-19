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
奖励函数配置数据模型和验证逻辑
提供完整的奖励函数参数配置、验证和管理能力
"""

import math
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class RewardWeights(BaseModel):
    """基础权重配置"""

    comfort: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="舒适度奖励权重 (0.0 - 2.0)"
    )
    pressure: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="压力惩罚权重 (0.0 - 2.0)"
    )
    static_penalty: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="静态时间惩罚权重 (0.0 - 2.0)"
    )
    energy: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="能量消耗惩罚权重 (0.0 - 2.0)"
    )

    @field_validator('*')
    @classmethod
    def check_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError('权重值不能为负数')
        return v

    @model_validator(mode='after')
    def validate_weight_sum(self) -> 'RewardWeights':
        total = self.comfort + self.pressure + self.static_penalty + self.energy
        if total < 0.5:
            raise ValueError(f'权重总和 ({total:.2f}) 过小，可能导致训练信号不足')
        if total > 5.0:
            raise ValueError(f'权重总和 ({total:.2f}) 过大，可能导致训练不稳定')
        return self


class ComfortConfig(BaseModel):
    """舒适度子项配置"""

    spine_alignment_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="脊柱对齐权重 (应与 pressure_uniformity_weight 之和为 1.0)"
    )
    pressure_uniformity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="压力均匀度权重 (应与 spine_alignment_weight 之和为 1.0)"
    )
    spine_curvature_sensitivity: float = Field(
        default=2.0,
        gt=0.0,
        le=10.0,
        description="脊柱曲率敏感度系数 (越大惩罚越严格)"
    )
    pressure_variance_sensitivity: float = Field(
        default=5.0,
        gt=0.0,
        le=20.0,
        description="压力方差敏感度系数 (越大惩罚越严格)"
    )

    @model_validator(mode='after')
    def validate_normalization(self) -> 'ComfortConfig':
        total = self.spine_alignment_weight + self.pressure_uniformity_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f'spine_alignment_weight + pressure_uniformity_weight 应等于 1.0，'
                f'当前值为 {total:.3f}'
            )
        return self


class ThresholdConfig(BaseModel):
    """阈值参数配置"""

    max_pressure: float = Field(
        default=50.0,
        gt=0.0,
        le=100.0,
        description="最大允许压力阈值"
    )
    static_duration: float = Field(
        default=900.0,
        gt=0.0,
        le=3600.0,
        description="静态时长阈值 (秒)，超过此时间开始惩罚"
    )
    action_magnitude_scale: float = Field(
        default=0.01,
        gt=0.0,
        le=0.1,
        description="动作幅度惩罚系数"
    )


class AdvancedConfig(BaseModel):
    """高级选项配置 (可选)"""

    enable_fatigue_awareness: bool = Field(
        default=True,
        description="是否启用疲劳感知惩罚"
    )
    fatigue_penalty_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="疲劳惩罚权重"
    )
    posture_change_reward: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="姿态变化奖励 (鼓励动态调整)"
    )
    symmetry_bonus: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="对称性奖励 (左右平衡)"
    )
    custom_formula: Optional[str] = Field(
        default=None,
        max_length=500,
        description="自定义奖励公式 (Python 表达式，可选)"
    )

    @field_validator('custom_formula')
    @classmethod
    def validate_custom_formula(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        
        v = v.strip()
        if not v:
            return None
        
        try:
            compile(v, '<formula>', 'eval')
        except SyntaxError as e:
            raise ValueError(f'公式语法错误: {e}')
        
        dangerous_patterns = [
            r'import\s',
            r'__\w+__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'\bos\b',
            r'\bsys\b',
            r'\bsubprocess\b',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('公式包含不安全的操作')
        
        return v


class ValidationResult(BaseModel):
    """配置验证结果"""

    valid: bool = Field(..., description="是否通过验证")
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="错误列表 [{field, issue, message}]"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="警告列表"
    )
    score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="配置质量评分 (0-100)"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="优化建议"
    )


class RewardConfig(BaseModel):
    """奖励函数完整配置模型"""

    config_name: Optional[str] = Field(
        None,
        max_length=100,
        description="配置名称"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="配置描述"
    )

    weights: RewardWeights = Field(
        default_factory=RewardWeights,
        description="基础权重配置"
    )
    comfort: ComfortConfig = Field(
        default_factory=ComfortConfig,
        description="舒适度子项配置"
    )
    thresholds: ThresholdConfig = Field(
        default_factory=ThresholdConfig,
        description="阈值参数配置"
    )
    advanced: Optional[AdvancedConfig] = Field(
        default=None,
        description="高级选项 (可选)"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="元数据 (作者、标签等)"
    )

    version: int = Field(
        default=1,
        ge=1,
        description="配置版本号"
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="创建时间"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="更新时间"
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.updated_at:
            self.updated_at = datetime.now()


class RewardBreakdown(BaseModel):
    """奖励值分解结果"""

    total_reward: float = Field(..., description="总奖励值")
    comfort_component: float = Field(..., description="舒适度分量")
    pressure_penalty: float = Field(..., description="压力惩罚分量")
    static_penalty: float = Field(..., description="静态惩罚分量")
    energy_penalty: float = Field(..., description="能量惩罚分量")
    fatigue_penalty: Optional[float] = Field(None, description="疲劳惩罚分量 (如果启用)")
    posture_bonus: Optional[float] = Field(None, description="姿态变化奖励 (如果启用)")
    symmetry_bonus: Optional[float] = Field(None, description="对称性奖励 (如果启用)")


class PresetInfo(BaseModel):
    """预设信息"""

    name: str = Field(..., description="预设名称")
    display_name: str = Field(..., description="显示名称")
    description: str = Field(..., description="预设描述")
    category: str = Field(default="general", description="分类")
    is_builtin: bool = Field(default=False, description="是否为内置预设")
    config: 'RewardConfig' = Field(..., description="预设的完整配置")


def get_default_config() -> RewardConfig:
    """
    获取默认奖励函数配置
    
    Returns:
        与 ChairEnv 硬编码参数一致的默认配置
    """
    return RewardConfig(
        config_name="default",
        description="默认均衡配置",
        weights=RewardWeights(
            comfort=1.0,
            pressure=0.8,
            static_penalty=0.5,
            energy=0.3
        ),
        comfort=ComfortConfig(
            spine_alignment_weight=0.5,
            pressure_uniformity_weight=0.5,
            spine_curvature_sensitivity=2.0,
            pressure_variance_sensitivity=5.0
        ),
        thresholds=ThresholdConfig(
            max_pressure=50.0,
            static_duration=900.0,
            action_magnitude_scale=0.01
        ),
        advanced=AdvancedConfig(
            enable_fatigue_awareness=True,
            fatigue_penalty_weight=0.2,
            posture_change_reward=0.05,
            symmetry_bonus=0.1
        ),
        metadata={
            "author: "Harley Wang (王华)",
            "tags": ["default", "balanced"]
        }
    )


BUILTIN_PRESETS: Dict[str, PresetInfo] = {}


def _initialize_presets():
    """初始化内置预设"""
    global BUILTIN_PRESETS

    presets_data = {
        "balanced": {
            "display_name": "均衡模式 (Balanced)",
            "description": "适用于大多数办公场景，在舒适度和健康之间取得平衡",
            "category": "general",
            "config": {
                "config_name": "balanced",
                "description": "均衡模式：舒适与健康兼顾",
                "weights": {"comfort": 1.0, "pressure": 0.8, "static_penalty": 0.5, "energy": 0.3},
                "comfort": {
                    "spine_alignment_weight": 0.5,
                    "pressure_uniformity_weight": 0.5,
                    "spine_curvature_sensitivity": 2.0,
                    "pressure_variance_sensitivity": 5.0
                },
                "thresholds": {"max_pressure": 50.0, "static_duration": 900.0, "action_magnitude_scale": 0.01}
            }
        },
        "health_first": {
            "display_name": "健康优先 (Health First)",
            "description": "侧重姿态纠正和健康监测，对不良坐姿零容忍",
            "category": "health",
            "config": {
                "config_name": "health_first",
                "description": "健康优先：严格纠正不良姿势",
                "weights": {"comfort": 0.6, "pressure": 1.5, "static_penalty": 1.2, "energy": 0.4},
                "comfort": {
                    "spine_alignment_weight": 0.7,
                    "pressure_uniformity_weight": 0.3,
                    "spine_curvature_sensitivity": 4.0,
                    "pressure_variance_sensitivity": 10.0
                },
                "thresholds": {"max_pressure": 35.0, "static_duration": 600.0, "action_magnitude_scale": 0.02}
            }
        },
        "comfort_priority": {
            "display_name": "舒适优先 (Comfort Priority)",
            "description": "最大化舒适度，容忍轻微的不良姿势，适合长时间工作",
            "category": "comfort",
            "config": {
                "config_name": "comfort_priority",
                "description": "舒适优先：最大化用户舒适体验",
                "weights": {"comfort": 1.8, "pressure": 0.4, "static_penalty": 0.2, "energy": 0.15},
                "comfort": {
                    "spine_alignment_weight": 0.35,
                    "pressure_uniformity_weight": 0.65,
                    "spine_curvature_sensitivity": 1.2,
                    "pressure_variance_sensitivity": 3.0
                },
                "thresholds": {"max_pressure": 70.0, "static_duration": 1800.0, "action_magnitude_scale": 0.005}
            }
        },
        "strict_posture": {
            "display_name": "严格工效学 (Strict Ergonomics)",
            "description": "严格的工效学标准，适合需要精确姿态控制的场景",
            "category": "ergonomics",
            "config": {
                "config_name": "strict_posture",
                "description": "严格工效学：零容忍不良坐姿",
                "weights": {"comfort": 0.8, "pressure": 1.8, "static_penalty": 1.5, "energy": 0.5},
                "comfort": {
                    "spine_alignment_weight": 0.85,
                    "pressure_uniformity_weight": 0.15,
                    "spine_curvature_sensitivity": 8.0,
                    "pressure_variance_sensitivity": 15.0
                },
                "thresholds": {"max_pressure": 30.0, "static_duration": 300.0, "action_magnitude_scale": 0.03}
            }
        },
        "energy_saving": {
            "display_name": "节能模式 (Energy Saving)",
            "description": "最小化座椅调整频率，适合低功耗设备或电池供电场景",
            "category": "efficiency",
            "config": {
                "config_name": "energy_saving",
                "description": "节能模式：减少座椅动作频率",
                "weights": {"comfort": 0.7, "pressure": 0.6, "static_penalty": 0.3, "energy": 1.5},
                "comfort": {
                    "spine_alignment_weight": 0.5,
                    "pressure_uniformity_weight": 0.5,
                    "spine_curvature_sensitivity": 1.5,
                    "pressure_variance_sensitivity": 4.0
                },
                "thresholds": {"max_pressure": 55.0, "static_duration": 1200.0, "action_magnitude_scale": 0.05}
            }
        }
    }

    for preset_name, data in presets_data.items():
        try:
            config = RewardConfig(**data["config"])
            BUILTIN_PRESETS[preset_name] = PresetInfo(
                name=preset_name,
                display_name=data["display_name"],
                description=data["description"],
                category=data.get("category", "general"),
                is_builtin=True,
                config=config
            )
        except Exception as e:
            print(f"Warning: Failed to initialize preset '{preset_name}': {e}")


_initialize_presets()


def get_preset(preset_name: str) -> Optional[RewardConfig]:
    """
    获取指定预设的配置副本
    
    Args:
        preset_name: 预设名称
        
    Returns:
        配置副本，如果预设不存在则返回 None
    """
    preset = BUILTIN_PRESETS.get(preset_name)
    if preset:
        return preset.config.model_copy(deep=True)
    return None


def list_presets() -> List[PresetInfo]:
    """
    列出所有可用的预设
    
    Returns:
        预设信息列表
    """
    return list(BUILTIN_PRESETS.values())


def validate_config(config_dict: Dict[str, Any]) -> ValidationResult:
    """
    验证配置字典是否合法
    
    Args:
        config_dict: 配置字典
        
    Returns:
        验证结果对象
    """
    errors = []
    warnings = []
    score = 100.0
    suggestions = []

    try:
        config = RewardConfig(**config_dict)
    except Exception as e:
        error_msg = str(e)
        
        field_match = re.search(r'(\w[\w.]*)', error_msg)
        if field_match:
            errors.append({
                "field": field_match.group(1),
                "issue": "validation_error",
                "message": error_msg
            })
        else:
            errors.append({
                "field": "unknown",
                "issue": "validation_error",
                "message": error_msg
            })
        
        return ValidationResult(
            valid=False,
            errors=errors,
            score=0.0
        )

    weights = config.weights
    weight_sum = weights.comfort + weights.pressure + weights.static_penalty + weights.energy
    
    if weight_sum < 1.0:
        warnings.append(f"权重总和较低 ({weight_sum:.2f})，可能导致训练收敛缓慢")
        score -= 10
        suggestions.append("建议增加各项权重的总和至 1.0 以上")

    if weight_sum > 3.5:
        warnings.append(f"权重总和较高 ({weight_sum:.2f})，可能导致梯度爆炸")
        score -= 10
        suggestions.append("建议降低各项权重，使总和保持在 1.0 - 3.0 范围内")

    comfort_weights = config.comfort.spine_alignment_weight + config.comfort.pressure_uniformity_weight
    if abs(comfort_weights - 1.0) > 0.001:
        errors.append({
            "field": "comfort",
            "issue": "normalization_error",
            "message": f"舒适度子项权重之和应为 1.0，当前为 {comfort_weights:.3f}"
        })
        score -= 20

    if config.thresholds.max_pressure < 25:
        warnings.append("最大压力阈值设置过低 (<25)，可能过于敏感")
        score -= 5
        suggestions.append("建议将 max_pressure 设置在 40-60 范围内")

    if config.thresholds.max_pressure > 80:
        warnings.append("最大压力阈值设置过高 (>80)，可能无法有效检测高压风险")
        score -= 5
        suggestions.append("建议将 max_pressure 设置在 40-60 范围内")

    if config.thresholds.static_duration < 300:
        warnings.append("静态时长阈值过短 (<5分钟)，可能导致频繁提醒")
        score -= 5
        suggestions.append("建议将 static_duration 设置在 600-1200 秒范围内")

    if config.advanced and config.advanced.custom_formula:
        suggestions.append("使用自定义公式时，请确保公式在安全沙箱中测试通过")

    balance_score = min(weights.comfort, weights.pressure) / max(weights.comfort, weights.pressure)
    if balance_score < 0.3:
        warnings.append("舒适度和压力权重严重不平衡")
        score -= 10
        suggestions.append("建议保持 comfort 和 pressure 权重在相近水平")

    score = max(0.0, min(100.0, score))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        score=score,
        suggestions=suggestions
    )


def calculate_reward_breakdown(
    config: RewardConfig,
    action_magnitude: float,
    spine_curvature: float,
    pressure_variance: float,
    max_pressure: float,
    static_duration: float,
    fatigue_level: float = 0.0,
    posture_changed: bool = False,
    symmetry_ratio: float = 1.0
) -> RewardBreakdown:
    """
    根据配置计算奖励值分解
    
    Args:
        config: 奖励函数配置
        action_magnitude: 动作幅度
        spine_curvature: 脊柱曲率偏差
        pressure_variance: 压力分布方差
        max_pressure: 最大压力点
        static_duration: 静态时长 (秒)
        fatigue_level: 疲劳程度 (0-1)
        posture_changed: 姿态是否发生变化
        symmetry_ratio: 左右对称比例 (0-1, 1.0 为完全对称)
    
    Returns:
        奖励值分解结果
    """
    w = config.weights
    c = config.comfort
    t = config.thresholds

    comfort_spine = c.spine_alignment_weight * math.exp(-c.spine_curvature_sensitivity * spine_curvature ** 2)
    comfort_pressure = c.pressure_uniformity_weight * math.exp(-c.pressure_variance_sensitivity * pressure_variance ** 2)
    comfort_total = w.comfort * (comfort_spine + comfort_pressure)

    pressure_penalty = 0.0
    if max_pressure > t.max_pressure:
        excess = max_pressure - t.max_pressure
        pressure_penalty = -w.pressure * (excess / t.max_pressure) ** 2

    static_penalty_val = 0.0
    if static_duration > t.static_duration:
        excess_time = static_duration - t.static_duration
        static_penalty_val = -w.static_penalty * (excess_time / t.static_duration) ** 1.5

    energy_penalty = -w.energy * t.action_magnitude_scale * action_magnitude ** 2

    total_reward = comfort_total + pressure_penalty + static_penalty_val + energy_penalty

    result = RewardBreakdown(
        total_reward=round(total_reward, 6),
        comfort_component=round(comfort_total, 6),
        pressure_penalty=round(pressure_penalty, 6),
        static_penalty=round(static_penalty_val, 6),
        energy_penalty=round(energy_penalty, 6)
    )

    if config.advanced:
        adv = config.advanced

        if adv.enable_fatigue_awareness and fatigue_level > 0:
            fatigue_p = -adv.fatigue_penalty_weight * fatigue_level ** 2
            result.fatigue_penalty = round(fatigue_p, 6)
            result.total_reward += fatigue_p

        if posture_changed and adv.posture_change_reward > 0:
            p_bonus = adv.posture_change_reward
            result.posture_bonus = round(p_bonus, 6)
            result.total_reward += p_bonus

        sym_deviation = abs(symmetry_ratio - 1.0)
        if sym_deviation < 0.3 and adv.symmetry_bonus > 0:
            s_bonus = adv.symmetry_bonus * (1.0 - sym_deviation / 0.3)
            result.symmetry_bonus = round(s_bonus, 6)
            result.total_reward += s_bonus

    result.total_reward = round(result.total_reward, 6)

    return result
