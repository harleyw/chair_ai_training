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

from fastapi import APIRouter, HTTPException
from typing import Dict
import logging

from api.models import (
    SensorData,
    AdjustmentRecommendation,
    ChairAction,
    BatchSensorData,
    BatchAdjustmentResponse,
    DetailedPostureAnalysis
)
from api.service import ChairAIService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chair", tags=["座椅控制"])


def get_service() -> ChairAIService:
    """获取服务实例（从 app state）"""
    from api.main import app
    return app.state.service


@router.post(
    "/adjust",
    response_model=AdjustmentRecommendation,
    summary="获取座椅调整建议",
    description="根据传感器数据返回最优的座椅调整建议"
)
async def get_adjustment(sensor_data: SensorData) -> AdjustmentRecommendation:
    """
    座椅调整建议接口
    
    - **pressure_matrix**: 8x8 压力传感器阵列数据
    - **posture_angles**: [头部, 肩部, 骨盆] 角度（度）
    - **sitting_duration**: 静坐时长（秒）
    - **user_weight**: 用户体重（kg）
    - **user_height**: 用户身高（m）
    - **fatigue_level**: 疲劳程度 (0-1)
    
    返回推荐的8维调整动作和详细分析。
    """
    try:
        service = get_service()
        
        data_dict = sensor_data.model_dump()
        
        action_vector, confidence = service.predict_action(data_dict)
        
        analysis, issues, comfort_score, risk_level = service.analyze_posture(data_dict)
        
        action = ChairAction(
            seat_height=round(float(action_vector[0]), 4),
            backrest_angle=round(float(action_vector[1]), 4),
            lumbar_position=round(float(action_vector[2]), 4),
            lumbar_thickness=round(float(action_vector[3]), 4),
            headrest_position=round(float(action_vector[4]), 4),
            headrest_angle=round(float(action_vector[5]), 4),
            left_armrest=round(float(action_vector[6]), 4),
            right_armrest=round(float(action_vector[7]), 4)
        )
        
        recommendations = issues[:5] if issues else ["当前姿态良好，保持现有设置"]
        
        if sensor_data.sitting_duration > 3600 and len(recommendations) < 5:
            recommendations.append("建议每小时起身活动5-10分钟")
        
        response = AdjustmentRecommendation(
            success=True,
            timestamp=__import__('datetime').datetime.now().isoformat(),
            action=action,
            confidence=round(float(confidence), 4),
            comfort_score=comfort_score,
            pressure_risk=risk_level,
            posture_analysis=analysis,
            recommendations=recommendations,
            posture_detail=self._build_posture_detail(analysis)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Adjustment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/batch-adjust",
    response_model=BatchAdjustmentResponse,
    summary="批量获取调整建议",
    description="一次处理多个时间点的传感器数据，返回批量调整建议"
)
async def batch_adjust(batch_data: BatchSensorData) -> BatchAdjustmentResponse:
    """批量推理接口"""
    import time
    
    start_time = time.time()
    service = get_service()
    
    results = []
    
    for i, sample in enumerate(batch_data.samples):
        try:
            data_dict = sample.model_dump()
            
            action_vector, confidence = service.predict_action(data_dict)
            
            analysis, issues, comfort_score, risk_level = service.analyze_posture(data_dict)
            
            action = ChairAction(
                seat_height=round(float(action_vector[0]), 4),
                backrest_angle=round(float(action_vector[1]), 4),
                lumbar_position=round(float(action_vector[2]), 4),
                lumbar_thickness=round(float(action_vector[3]), 4),
                headrest_position=round(float(action_vector[4]), 4),
                headrest_angle=round(float(action_vector[5]), 4),
                left_armrest=round(float(action_vector[6]), 4),
                right_armrest=round(float(action_vector[7]), 4)
            )
            
            result = AdjustmentRecommendation(
                success=True,
                timestamp=__import__('datetime').datetime.now().isoformat(),
                action=action,
                confidence=round(float(confidence), 4),
                comfort_score=comfort_score,
                pressure_risk=risk_level,
                posture_analysis=analysis,
                recommendations=issues[:3] if issues else ["正常"],
                posture_detail=_build_posture_detail(analysis)
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Sample {i} error: {e}")
            results.append(AdjustmentRecommendation(
                success=False,
                timestamp="",
                action=ChairAction(**{k: 0.0 for k in ChairAction.model_fields}),
                confidence=0.0,
                comfort_score=0.0,
                pressure_risk="error",
                posture_analysis={},
                recommendations=[f"处理错误: {str(e)}"]
            ))
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchAdjustmentResponse(
        success=True,
        total_samples=len(results),
        results=results,
        processing_time_ms=round(processing_time, 2)
    )


@router.get("/demo", summary="演示接口", description="返回示例传感器数据和调整结果")
async def demo_adjustment() -> AdjustmentRecommendation:
    """演示接口，返回预设的示例数据"""
    demo_data = SensorData(
        pressure_matrix=[
            [0.1, 0.15, 0.25, 0.35, 0.32, 0.28, 0.18, 0.12],
            [0.12, 0.18, 0.30, 0.45, 0.42, 0.35, 0.20, 0.14],
            [0.08, 0.12, 0.22, 0.38, 0.36, 0.28, 0.16, 0.10],
            [0.05, 0.08, 0.15, 0.28, 0.26, 0.20, 0.12, 0.07],
            [0.04, 0.06, 0.12, 0.22, 0.20, 0.15, 0.09, 0.05],
            [0.03, 0.05, 0.10, 0.18, 0.16, 0.12, 0.07, 0.04],
            [0.02, 0.03, 0.07, 0.14, 0.12, 0.09, 0.05, 0.03],
            [0.01, 0.02, 0.05, 0.10, 0.08, 0.06, 0.03, 0.02]
        ],
        posture_angles=[22.0, -12.0, 15.0],
        sitting_duration=2700.0,
        user_weight=72.0,
        user_height=1.68,
        fatigue_level=0.55
    )
    
    return await get_adjustment(demo_data)


@router.post("/quick-adjust", summary="快速调整接口", description="简化版接口，仅需要基本参数")
async def quick_adjust(
    sitting_minutes: float = 30.0,
    discomfort_level: float = 0.5,
    body_type: str = "average"
) -> Dict:
    """
    快速调整接口
    
    参数:
    - sitting_minutes: 已坐时间（分钟）
    - discomfort_level: 不适程度 (0-1)
    - body_type: 体型 (thin/average/heavy)
    """
    body_params = {
        "thin": {"weight": 55, "height": 1.65},
        "average": {"weight": 70, "height": 1.70},
        "heavy": {"weight": 90, "height": 1.75}
    }
    
    params = body_params.get(body_type, body_params["average"])
    
    fatigue = min(discomfort_level * 0.8 + (sitting_minutes / 120) * 0.2, 1.0)
    
    sensor_data = SensorData(
        pressure_matrix=[[0.2] * 8 for _ in range(8)],
        posture_angles=[discomfort_level * 20, -discomfort_level * 15, discomfort_level * 12],
        sitting_duration=sitting_minutes * 60,
        user_weight=params["weight"],
        user_height=params["height"],
        fatigue_level=fatigue
    )
    
    result = await get_adjustment(sensor_data)
    return result.model_dump()


def _build_posture_detail(analysis: dict) -> Optional[DetailedPostureAnalysis]:
    """
    从 analysis 字典中构建 DetailedPostureAnalysis 对象

    Args:
        analysis: 包含 posture_detail 的分析字典

    Returns:
        DetailedPostureAnalysis 实例或 None
    """
    posture_detail = analysis.get("posture_detail")

    if not posture_detail or not isinstance(posture_detail, dict):
        return None

    try:
        return DetailedPostureAnalysis(
            posture_type=posture_detail.get("posture_type", "unknown"),
            posture_name_cn=posture_detail.get("posture_name_cn", "未知"),
            severity=posture_detail.get("severity", "good"),
            confidence=posture_detail.get("confidence", 0.0),
            risk_areas=posture_detail.get("risk_areas", []),
            recommended_exercises=posture_detail.get("recommended_exercises", []),
            primary_adjustments=posture_detail.get("primary_adjustments", {}),
            message=posture_detail.get("message", "")
        )
    except Exception as e:
        logger.error(f"Error building posture detail: {e}")
        return None
