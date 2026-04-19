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
奖励函数配置 REST API 路由
提供配置管理、预设操作、预览计算等端点
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

from api.reward_config import (
    RewardConfig,
    RewardWeights,
    ComfortConfig,
    ThresholdConfig,
    AdvancedConfig,
    ValidationResult,
    RewardBreakdown,
    PresetInfo,
    get_default_config,
    get_preset,
    list_presets,
    validate_config,
    calculate_reward_breakdown,
    BUILTIN_PRESETS
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/reward",
    tags=["奖励函数配置"]
)

_current_config: RewardConfig = get_default_config()
_custom_presets: Dict[str, RewardConfig] = {}
_config_version: int = 1


@router.get("/config", response_model=RewardConfig)
async def get_current_config():
    """
    获取当前生效的奖励函数配置
    
    返回当前内存中的完整配置，包括元数据和版本信息。
    """
    global _current_config
    return _current_config


@router.put("/config", response_model=RewardConfig)
async def update_config(config: RewardConfig):
    """
    更新奖励函数配置
    
    验证并保存新的奖励函数配置。验证失败时返回详细错误信息。
    
    - **config**: 完整的奖励函数配置对象
    
    Returns:
        保存后的配置（包含更新后的版本号）
    """
    global _current_config, _config_version

    validation = validate_config(config.model_dump())
    if not validation.valid:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "INVALID_CONFIG",
                "message": "配置验证失败",
                "errors": validation.errors,
                "warnings": validation.warnings,
                "score": validation.score,
                "suggestions": validation.suggestions
            }
        )

    if validation.warnings:
        logger.warning(f"Config validation warnings: {validation.warnings}")

    _config_version += 1
    config.version = _config_version
    config.updated_at = datetime.now()
    _current_config = config

    logger.info(f"Reward config updated to version {_config_version}")
    return _current_config


@router.post("/config/validate", response_model=ValidationResult)
async def validate_config_endpoint(config: RewardConfig):
    """
    验证奖励函数配置（不保存）
    
    仅对提交的配置进行验证，返回详细的验证结果，但不保存到系统。
    可用于在正式应用前预检查配置的合法性。
    
    - **config**: 待验证的配置对象
    
    Returns:
        验证结果，包含通过/失败状态、错误列表、警告和优化建议
    """
    result = validate_config(config.model_dump())
    return result


@router.delete("/config", response_model=RewardConfig)
async def reset_config():
    """
    重置为默认配置
    
    将当前配置重置为系统默认值。
    
    Returns:
        重置后的默认配置
    """
    global _current_config, _config_version

    _config_version += 1
    _current_config = get_default_config()
    _current_config.version = _config_version
    _current_config.updated_at = datetime.now()

    logger.info("Reward config reset to default")
    return _current_config


@router.get("/presets", response_model=List[PresetInfo])
async def list_all_presets(
    category: Optional[str] = Query(None, description="按分类筛选"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量")
):
    """
    列出所有可用的奖励函数预设
    
    返回内置预设和用户自定义预设的列表。支持分页和分类筛选。
    
    - **category**: 可选的分类筛选 (general/health/comfort/ergonomics/efficiency)
    - **page**: 页码 (从 1 开始)
    - **size**: 每页数量
    
    Returns:
        预设信息列表
    """
    all_presets = list_presets()

    for name, config in _custom_presets.items():
        all_presets.append(PresetInfo(
            name=name,
            display_name=config.config_name or name,
            description=config.description or "",
            category="custom",
            is_builtin=False,
            config=config
        ))

    if category:
        all_presets = [p for p in all_presets if p.category == category]

    start = (page - 1) * size
    end = start + size
    return all_presets[start:end]


@router.get("/presets/{preset_name}", response_model=PresetInfo)
async def get_preset_detail(preset_name: str):
    """
    获取指定预设的详细信息
    
    - **preset_name**: 预设名称
    
    Returns:
        预设完整信息，包含配置详情
    """
    builtin_preset = BUILTIN_PRESETS.get(preset_name)
    if builtin_preset:
        return builtin_preset

    custom_preset = _custom_presets.get(preset_name)
    if custom_preset:
        return PresetInfo(
            name=preset_name,
            display_name=custom_preset.config_name or preset_name,
            description=custom_preset.description or "",
            category="custom",
            is_builtin=False,
            config=custom_preset
        )

    raise HTTPException(
        status_code=404,
        detail=f"预设 '{preset_name}' 不存在"
    )


@router.post("/presets", response_model=PresetInfo)
async def create_custom_preset(config: RewardConfig):
    """
    创建自定义预设
    
    将当前配置保存为一个自定义预设。config_name 必须唯一。
    
    - **config**: 要保存为预设的配置 (需提供 config_name)
    
    Returns:
        创建成功的预设信息
    """
    if not config.config_name:
        raise HTTPException(
            status_code=400,
            detail="创建自定义预设必须提供 config_name"
        )

    preset_name = config.config_name.lower().replace(" ", "_")

    if preset_name in BUILTIN_PRESETS:
        raise HTTPException(
            status_code=409,
            detail=f"预设名称 '{preset_name}' 与内置预设冲突，请使用其他名称"
        )

    if preset_name in _custom_presets:
        raise HTTPException(
            status_code=409,
            detail=f"自定义预设 '{preset_name}' 已存在，请使用 PUT 接口更新"
        )

    validation = validate_config(config.model_dump())
    if not validation.valid:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "INVALID_CONFIG",
                "message": "配置验证失败",
                "errors": validation.errors
            }
        )

    _custom_presets[preset_name] = config

    logger.info(f"Custom preset created: {preset_name}")
    return PresetInfo(
        name=preset_name,
        display_name=config.config_name,
        description=config.description or "",
        category="custom",
        is_builtin=False,
        config=config
    )


@router.put("/presets/{preset_name}", response_model=PresetInfo)
async def update_custom_preset(preset_name: str, config: RewardConfig):
    """
    更新自定义预设
    
    只能更新用户创建的自定义预设，不能修改内置预设。
    
    - **preset_name**: 预设名称
    - **config**: 新的配置内容
    
    Returns:
        更新后的预设信息
    """
    if preset_name in BUILTIN_PRESETS:
        raise HTTPException(
            status_code=403,
            detail=f"不允许修改内置预设 '{preset_name}'"
        )

    if preset_name not in _custom_presets:
        raise HTTPException(
            status_code=404,
            detail=f"自定义预设 '{preset_name}' 不存在"
        )

    validation = validate_config(config.model_dump())
    if not validation.valid:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "INVALID_CONFIG",
                "message": "配置验证失败",
                "errors": validation.errors
            }
        )

    config.version = _custom_presets[preset_name].version + 1
    config.updated_at = datetime.now()
    _custom_presets[preset_name] = config

    logger.info(f"Custom preset updated: {preset_name}")
    return PresetInfo(
        name=preset_name,
        display_name=config.config_name or preset_name,
        description=config.description or "",
        category="custom",
        is_builtin=False,
        config=config
    )


@router.delete("/presets/{preset_name}", response_model=dict)
async def delete_custom_preset(preset_name: str):
    """
    删除自定义预设
    
    只能删除用户创建的自定义预设，不能删除内置预设。
    
    - **preset_name**: 预设名称
    
    Returns:
        操作结果
    """
    if preset_name in BUILTIN_PRESETS:
        raise HTTPException(
            status_code=403,
            detail=f"不允许删除内置预设 '{preset_name}'"
        )

    if preset_name not in _custom_presets:
        raise HTTPException(
            status_code=404,
            detail=f"自定义预设 '{preset_name}' 不存在"
        )

    del _custom_presets[preset_name]

    logger.info(f"Custom preset deleted: {preset_name}")
    return {
        "success": True,
        "message": f"预设 '{preset_name}' 已成功删除"
    }


@router.post("/presets/{preset_name}/apply", response_model=RewardConfig)
async def apply_preset(preset_name: str):
    """
    应用预设到当前配置
    
    将指定的预设配置应用到当前生效的配置中。
    
    - **preset_name**: 要应用的预设名称
    
    Returns:
        应用后的当前配置
    """
    global _current_config, _config_version

    preset = get_preset(preset_name)
    if not preset and preset_name in _custom_presets:
        preset = _custom_presets[preset_name].model_copy(deep=True)

    if not preset:
        raise HTTPException(
            status_code=404,
            detail=f"预设 '{preset_name}' 不存在"
        )

    _config_version += 1
    preset.version = _config_version
    preset.updated_at = datetime.now()
    _current_config = preset

    logger.info(f"Preset applied: {preset_name} -> version {_config_version}")
    return _current_config


class PreviewCalculateRequest(BaseModel):
    """单点计算请求"""
    config: Optional[RewardConfig] = Field(None, description="使用的配置 (可选，默认使用当前配置)")
    action_magnitude: float = Field(..., gt=0, description="动作幅度")
    spine_curvature: float = Field(0.0, description="脊柱曲率偏差")
    pressure_variance: float = Field(0.0, description="压力分布方差")
    max_pressure: float = Field(30.0, gt=0, description="最大压力点")
    static_duration: float = Field(0.0, ge=0, description="静态时长 (秒)")
    fatigue_level: float = Field(0.0, ge=0, le=1, description="疲劳程度 (0-1)")
    posture_changed: bool = Field(False, description="姿态是否发生变化")
    symmetry_ratio: float = Field(1.0, gt=0, le=1, description="左右对称比例")


class CurveDataPoint(BaseModel):
    """曲线数据点"""
    x: float = Field(..., description="自变量值")
    y: float = Field(..., description="因变量值 (奖励值)")


class PreviewCurveRequest(BaseModel):
    """曲线生成请求"""
    config: Optional[RewardConfig] = Field(None, description="使用的配置")
    variable_name: str = Field(..., description="变化的变量名")
    variable_range: tuple = Field((0.0, 1.0), description="变量取值范围 [min, max]")
    num_points: int = Field(50, ge=10, le=500, description="采样点数")
    fixed_params: Dict[str, float] = Field(default_factory=dict, description="固定参数")


class CompareRequest(BaseModel):
    """多配置对比请求"""
    configs: List[RewardConfig] = Field(..., min_length=2, description="要对比的配置列表 (至少2个)")
    sensor_data: PreviewCalculateRequest = Field(..., description="传感器数据")


@router.post("/preview/calculate", response_model=RewardBreakdown)
async def preview_calculate(request: PreviewCalculateRequest):
    """
    单点奖励值计算
    
    使用指定配置（或当前配置）计算给定传感器数据下的奖励值分解。
    
    - **request**: 包含配置和传感器数据的请求体
    
    Returns:
        奖励值分解结果 (总奖励、各分量明细)
    """
    config = request.config if request.config else _current_config

    try:
        breakdown = calculate_reward_breakdown(
            config=config,
            action_magnitude=request.action_magnitude,
            spine_curvature=request.spine_curvature,
            pressure_variance=request.pressure_variance,
            max_pressure=request.max_pressure,
            static_duration=request.static_duration,
            fatigue_level=request.fatigue_level,
            posture_changed=request.posture_changed,
            symmetry_ratio=request.symmetry_ratio
        )
        return breakdown
    except Exception as e:
        logger.error(f"Preview calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview/curve", response_model=List[CurveDataPoint])
async def preview_curve(request: PreviewCurveRequest):
    """
    生成奖励函数曲线数据
    
    在指定变量的取值范围内，计算对应的奖励值变化曲线。
    可用于可视化展示参数对奖励的影响。
    
    - **request**: 曲线生成参数
    
    Returns:
        曲线数据点数组 [(x1,y1), (x2,y2), ...]
    """
    config = request.config if request.config else _current_config
    var_name = request.variable_name
    var_min, var_max = request.variable_range
    num_points = request.num_points
    fixed = request.fixed_params

    valid_vars = {
        'action_magnitude', 'spine_curvature', 'pressure_variance',
        'max_pressure', 'static_duration', 'fatigue_level', 'symmetry_ratio'
    }

    if var_name not in valid_vars:
        raise HTTPException(
            status_code=400,
            detail=f"无效的变量名 '{var_name}'，有效选项: {valid_vars}"
        )

    points = []
    step = (var_max - var_min) / (num_points - 1)

    for i in range(num_points):
        x_val = var_min + i * step
        
        params = {
            'action_magnitude': fixed.get('action_magnitude', 0.5),
            'spine_curvature': fixed.get('spine_curvature', 0.3),
            'pressure_variance': fixed.get('pressure_variance', 0.2),
            'max_pressure': fixed.get('max_pressure', 35.0),
            'static_duration': fixed.get('static_duration', 600.0),
            'fatigue_level': fixed.get('fatigue_level', 0.3),
            'posture_changed': False,
            'symmetry_ratio': fixed.get('symmetry_ratio', 0.9)
        }
        params[var_name] = x_val

        try:
            breakdown = calculate_reward_breakdown(config=config, **params)
            points.append(CurveDataPoint(x=x_val, y=breakdown.total_reward))
        except Exception as e:
            logger.warning(f"Curve point calculation error at x={x_val}: {e}")

    return points


@router.post("/preview/compare", response_model=List[Dict[str, Any]])
async def preview_compare(request: CompareRequest):
    """
    多配置对比分析
    
    对比多个配置在同一组传感器数据下的表现差异。
    
    - **request**: 包含多个配置和传感器数据的请求体
    
    Returns:
        对比表格，每个配置的各分量奖励值
    """
    results = []

    for idx, config in enumerate(request.configs):
        try:
            breakdown = calculate_reward_breakdown(
                config=config,
                action_magnitude=request.sensor_data.action_magnitude,
                spine_curvature=request.sensor_data.spine_curvature,
                pressure_variance=request.sensor_data.pressure_variance,
                max_pressure=request.sensor_data.max_pressure,
                static_duration=request.sensor_data.static_duration,
                fatigue_level=request.sensor_data.fatigue_level,
                posture_changed=request.sensor_data.posture_changed,
                symmetry_ratio=request.sensor_data.symmetry_ratio
            )
            
            results.append({
                "config_index": idx,
                "config_name": config.config_name or f"Configuration_{idx}",
                "total_reward": breakdown.total_reward,
                "comfort_component": breakdown.comfort_component,
                "pressure_penalty": breakdown.pressure_penalty,
                "static_penalty": breakdown.static_penalty,
                "energy_penalty": breakdown.energy_penalty,
                "fatigue_penalty": breakdown.fatigue_penalty,
                "posture_bonus": breakdown.posture_bonus,
                "symmetry_bonus": breakdown.symmetry_bonus
            })
        except Exception as e:
            results.append({
                "config_index": idx,
                "config_name": config.config_name or f"Configuration_{idx}",
                "error": str(e)
            })

    return results


class ExportResponse(BaseModel):
    """导出响应"""
    success: bool
    format: str
    data: Dict[str, Any]
    exported_at: str


@router.get("/export", response_model=ExportResponse)
async def export_config(format: str = Query("json", regex="^(json|yaml)$")):
    """
    导出当前配置
    
    将当前奖励函数配置导出为 JSON 或 YAML 格式。
    
    - **format**: 导出格式 (json 或 yaml)
    
    Returns:
        导出的配置数据
    """
    global _current_config

    export_data = _current_config.model_dump()

    export_data["metadata"] = {
        "exported_by": "chair_ai_api",
        "export_format": format,
        "api_version": "2.2.0",
        "exported_at": datetime.now().isoformat(),
        "source": "reward_config_export"
    }

    return ExportResponse(
        success=True,
        format=format,
        data=export_data,
        exported_at=datetime.now().isoformat()
    )


class ImportResponse(BaseModel):
    """导入响应"""
    success: bool
    message: str
    config: Optional[RewardConfig]
    validation: Optional[ValidationResult]


@router.post("/import", response_model=ImportResponse)
async def import_config(data: Dict[str, Any]):
    """
    导入配置
    
    从 JSON 或 YAML 格式的字典导入奖励函数配置。
    导入时会自动进行验证，但不会直接应用到当前配置。
    
    - **data**: 配置字典
    
    Returns:
        导入结果，包含验证状态和导入后的配置对象
    """
    try:
        config = RewardConfig(**data)
    except Exception as e:
        return ImportResponse(
            success=False,
            message=f"配置解析失败: {str(e)}",
            config=None,
            validation=None
        )

    validation = validate_config(data)

    return ImportResponse(
        success=validation.valid,
        message="配置导入成功" if validation.valid else f"配置导入成功但有 {len(validation.errors)} 个错误",
        config=config,
        validation=validation
    )


class ConfigStatus(BaseModel):
    """配置状态"""
    current_config_name: Optional[str]
    version: int
    is_modified: bool
    last_modified: Optional[str]
    total_custom_presets: int
    total_builtin_presets: int
    available_categories: List[str]


@router.get("/status", response_model=ConfigStatus)
async def get_config_status():
    """
    获取配置状态查询
    
    返回当前配置系统的状态信息，包括版本、是否已修改等。
    
    Returns:
        配置状态信息
    """
    global _current_config, _config_version

    categories = set()
    for preset in list_presets():
        categories.add(preset.category)

    return ConfigStatus(
        current_config_name=_current_config.config_name,
        version=_config_version,
        is_modified=_current_config.config_name != "default",
        last_modified=_current_config.updated_at.isoformat() if _current_config.updated_at else None,
        total_custom_presets=len(_custom_presets),
        total_builtin_presets=len(BUILTIN_PRESETS),
        available_categories=list(categories)
    )


class DiffItem(BaseModel):
    """差异项"""
    path: str
    type: str = Field(..., pattern="^(added|modified|removed)$")
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None


class DiffResponse(BaseModel):
    """差异对比响应"""
    base_config_name: str
    new_config_name: str
    differences: List[DiffItem]
    summary: Dict[str, int]


@router.post("/diff", response_model=DiffResponse)
async def compare_configs_diff(base_config: RewardConfig, new_config: RewardConfig):
    """
    配置差异对比
    
    比较两个配置之间的差异，返回新增、修改、删除的字段。
    
    - **base_config**: 基础配置 (旧版本)
    - **new_config**: 新配置 (新版本)
    
    Returns:
        差异对比结果
    """
    base_dict = base_config.model_dump()
    new_dict = new_config.model_dump()

    differences = []
    added_count = 0
    modified_count = 0
    removed_count = 0

    def compare_dicts(base: dict, new: dict, prefix: str = ""):
        nonlocal added_count, modified_count, removed_count

        all_keys = set(base.keys()) | set(new.keys())

        for key in sorted(all_keys):
            path = f"{prefix}.{key}" if prefix else key

            if key not in base:
                differences.append(DiffItem(path=path, type="added", new_value=new[key]))
                added_count += 1
            elif key not in new:
                differences.append(DiffItem(path=path, type="removed", old_value=base[key]))
                removed_count += 1
            else:
                old_val = base[key]
                new_val = new[key]

                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    compare_dicts(old_val, new_val, path)
                elif old_val != new_val:
                    differences.append(DiffItem(
                        path=path,
                        type="modified",
                        old_value=old_val,
                        new_value=new_val
                    ))
                    modified_count += 1

    compare_dicts(base_dict, new_dict)

    return DiffResponse(
        base_config_name=base_config.config_name or "unnamed",
        new_config_name=new_config.config_name or "unnamed",
        differences=differences,
        summary={"added": added_count, "modified": modified_count, "removed": removed_count}
    )


class ScoreResponse(BaseModel):
    """评分响应"""
    score: float
    grade: str
    suggestions: List[str]


@router.post("/score", response_model=ScoreResponse)
async def score_config(config: RewardConfig):
    """
    配置评分与建议
    
    对提交的配置进行质量评分 (0-100)，并提供优化建议。
    
    - **config**: 待评分的配置
    
    Returns:
        评分结果和建议列表
    """
    validation = validate_config(config.model_dump())

    score = validation.score

    if score >= 90:
        grade = "A (优秀)"
    elif score >= 75:
        grade = "B (良好)"
    elif score >= 60:
        grade = "C (合格)"
    elif score >= 40:
        grade = "D (需改进)"
    else:
        grade = "F (不合格)"

    return ScoreResponse(
        score=score,
        grade=grade,
        suggestions=validation.suggestions
    )
