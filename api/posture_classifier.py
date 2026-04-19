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
高级姿态分类器
支持 8 种具体坐姿类型的自动识别和严重程度评估

支持的姿态类型:
1. normal - 正常坐姿
2. forward_lean - 前倾/探头
3. backward_recline - 后仰/瘫坐
4. lateral_tilt - 左偏/右偏
5. crossed_legs - 交叉腿坐
6. leg_crossed - 跷二郎腿
7. lotus_position - 盘腿坐
8. forward_reach - 前伸坐姿
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


class PostureType(Enum):
    """姿态类型枚举"""
    NORMAL = "normal"
    FORWARD_LEAN = "forward_lean"
    BACKWARD_RECLINE = "backward_recline"
    LATERAL_TILT = "lateral_tilt"
    CROSSED_LEGS = "crossed_legs"
    LEG_CROSSED = "leg_crossed"
    LOTUS_POSITION = "lotus_position"
    FORWARD_REACH = "forward_reach"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """严重程度枚举"""
    IDEAL = "ideal"
    GOOD = "good"
    WARNING = "warning"
    DANGER = "danger"


@dataclass
class PressureFeatures:
    """压力矩阵派生特征"""
    left_right_balance: float = 0.0       # 左右平衡 (-1=全左, +1=全右)
    front_back_balance: float = 0.0        # 前后平衡 (-1=全前, +1=全后)
    asymmetry_index: float = 0.0           # 不对称指数 (0=对称, 1=完全不对称)
    center_of_pressure: Tuple[float, float] = (0.5, 0.5)  # 压力重心 (x, y) 归一化坐标
    max_pressure: float = 0.0              # 最大压力值
    mean_pressure: float = 0.0             # 平均压力值
    left_ratio: float = 0.5                # 左侧占比
    right_ratio: float = 0.5               # 右侧占比
    front_ratio: float = 0.5               # 前部占比
    back_ratio: float = 0.5                # 后部占比


@dataclass
class PostureResult:
    """姿态分类结果"""
    posture_type: PostureType
    severity: SeverityLevel
    confidence: float                       # 分类置信度 (0-1)
    
    # 详细信息
    risk_areas: List[str] = field(default_factory=list)
    recommended_exercises: List[str] = field(default_factory=list)
    primary_adjustments: Dict[str, float] = field(default_factory=dict)
    message: str = ""
    
    # 原始特征（用于调试）
    raw_angles: Dict[str, float] = field(default_factory=dict)
    pressure_features: Optional[PressureFeatures] = None


# ========== 姿态建议策略常量 ==========

POSTURE_ADJUSTMENT_STRATEGY: Dict[PostureType, Dict[str, Any]] = {
    PostureType.NORMAL: {
        "severity": SeverityLevel.IDEAL,
        "message": "当前坐姿良好，请继续保持",
        "risk_areas": [],
        "exercises": [],
        "adjustments": {},
        "primary_actions": {}
    },
    
    PostureType.FORWARD_LEAN: {
        "severity": SeverityLevel.WARNING,
        "message": "检测到前倾/探头姿势，建议：①将座垫后移 2-3cm ②调直靠背至90-100° ③显示器抬高至视线水平",
        "risk_areas": ["颈椎 (C4-C7)", "肩胛提肌", "斜方肌"],
        "exercises": ["收下巴练习(每次5秒×10次)", "胸椎伸展运动", "肩胛骨回缩练习"],
        "adjustments": {
            "seat_height": 0.1,
            "backrest_angle": -0.15,
            "headrest_position": 0.2,
            "headrest_angle": -0.1
        },
        "primary_actions": {"seat_height": "+10%", "backrest_angle": "-15%"}
    },
    
    PostureType.BACKWARD_RECLINE: {
        "severity": SeverityLevel.WARNING,
        "message": "检测到后仰/瘫坐姿势，建议：①增强腰托支撑厚度 ②调整靠背角度至100-110° ③双脚平放地面",
        "risk_areas": ["腰椎 (L3-L5)", "髂腰肌", "臀部肌肉"],
        "exercises": ["核心激活训练(平板支撑)", "臀桥运动(15次×3组)", "髂腰肌拉伸"],
        "adjustments": {
            "lumbar_thickness": 0.35,
            "backrest_angle": 0.12,
            "seat_height": -0.05
        },
        "primary_actions": {"lumbar_thickness": "+35%", "backrest_angle": "+12%"}
    },
    
    PostureType.LATERAL_TILT: {
        "severity": SeverityLevel.WARNING,
        "message": "检测到{direction}偏坐姿势，建议：①调整{opposite}扶手高度 ②身体向{opposite}侧微移 ③检查座椅是否水平",
        "risk_areas": ["脊柱侧方弯曲", "骶髂关节", "{side}侧腰肌群"],
        "exercises": ["脊柱侧弯矫正伸展", "髋关节环绕活动", "{side}侧腰肌群拉伸"],
        "adjustments": {
            "left_armrest": 0.25,
            "right_armrest": -0.25,
            "lumbar_position": 0.15
        },
        "primary_actions": {"armrests": "平衡调整"}
    },
    
    PostureType.CROSSED_LEGS: {
        "severity": SeverityLevel.WARNING,
        "message": "检测到交叉腿坐姿，建议：②双脚平放地面 ②膝盖保持与臀部同高 ③避免长时间维持同一姿势",
        "risk_areas": ["骨盆旋转", "髋关节", "膝关节内侧副韧带"],
        "exercises": ["髋关节灵活性训练", "内收肌群拉伸", "臀部肌肉放松"],
        "adjustments": {
            "seat_height": 0.08,
            "lumbar_position": 0.1
        },
        "primary_actions": {"seat_height": "+8%", "uncross_legs": "请伸直双腿"}
    },
    
    PostureType.LEG_CROSSED: {
        "severity": SeverityLevel.DANGER,
        "message": "⚠️ 检测到跷二郎腿姿势！建议：①立即放下翘起的腿 ②双脚平放地面 ③此姿势会导致骨盆倾斜和静脉回流受阻",
        "risk_areas": ["骨盆倾斜", "静脉受压(血栓风险)", "{dominant_side}侧坐骨神经"],
        "exercises": ["腘绳肌拉伸", "臀部肌肉放松", "踝关节泵式运动"],
        "adjustments": {
            "seat_height": 0.05,
            "lumbar_thickness": 0.2,
            f"{['left', 'right'][0]}_armrest": 0.15
        },
        "primary_actions": {"uncross_leg": "⚠️ 请立即放下翘起的腿"}
    },
    
    PostureType.LOTUS_POSITION: {
        "severity": SeverityLevel.GOOD,
        "message": "检测到盘腿坐姿，此姿势对髋关节有一定压力但可短期维持，建议每30分钟切换为正常坐姿",
        "risk_areas": ["髋关节外展肌", "膝关节外侧"],
        "exercises": ["髋关节内收训练", "股四头肌拉伸", "臀部滚动放松"],
        "adjustments": {
            "seat_width_adjust": 0.1,
            "lumbar_position": 0.05
        },
        "primary_actions": {"switch_every_30min": "建议定时切换姿势"}
    },
    
    PostureType.FORWARD_REACH: {
        "severity": SeverityLevel.DANGER,
        "message": "🔴 检测到前伸坐姿！身体远离靠背导致腰部完全失去支撑，强烈建议：①向后移动靠近靠背 ②调整桌椅距离 ③使用腰部支撑垫",
        "risk_areas": ["腰椎间盘(全部节段)", "竖脊肌(过度牵拉)", "腹部压力(消化影响)"],
        "exercises": ["猫牛式脊柱活动", "婴儿式放松", "死虫式核心激活"],
        "adjustments": {
            "lumbar_thickness": 0.4,
            "backrest_angle": 0.2,
            "seat_height": -0.08
        },
        "primary_actions": {"move_closer_to_backrest": "🔴 请向后移动靠近靠背!"}
    },

    PostureType.UNKNOWN: {
        "severity": SeverityLevel.WARNING,
        "message": "无法识别当前姿态类型，请检查传感器数据是否正常",
        "risk_areas": ["未知"],
        "exercises": [],
        "adjustments": {},
        "primary_actions": {}
    }
}


class PostureClassifier:
    """
    高级姿态分类器
    
    使用基于规则的专家系统进行多维度特征融合判断，
    支持 8 种具体坐姿类型和 4 级严重程度评估。
    """
    
    def __init__(self):
        self.angle_thresholds = {
            'head_normal': 15.0,
            'head_slight': 25.0,
            'shoulder_normal': 10.0,
            'shoulder_slight': 20.0,
            'pelvis_normal': 8.0,
            'pelvis_slight': 15.0,
            'recline_threshold': 20.0,
            'lateral_threshold': 12.0,
        }
        
        self.pressure_thresholds = {
            'asymmetry_mild': 0.15,
            'asymmetry_moderate': 0.25,
            'asymmetry_severe': 0.40,
            'single_side_dominance': 0.65,
            'front_reach_threshold': 0.35,
            'outer_spread_ratio': 0.55,
        }
    
    def classify(
        self,
        sensor_data: Dict[str, Any],
        history: Optional[List[Dict]] = None
    ) -> PostureResult:
        """
        主分类方法
        
        Args:
            sensor_data: 传感器数据字典（需包含 posture_angles 和 pressure_matrix）
            history: 可选的历史数据列表
            
        Returns:
            PostureResult: 完整的分类结果
        """
        # Step 1: 提取角度特征
        angles = self._extract_angles(sensor_data)
        
        # Step 2: 计算压力特征
        pressure_features = self._analyze_pressure_matrix(sensor_data)
        
        # Step 3: 规则匹配
        posture_type = self._match_rules(angles, pressure_features, history)
        
        # Step 4: 计算严重程度
        severity = self._calculate_severity(angles, pressure_features, posture_type)
        
        # Step 5: 生成建议策略
        strategy = POSTURE_ADJUSTMENT_STRATEGY.get(posture_type, 
                                                POSTURE_ADJUSTMENT_STRATEGY[PostureType.UNKNOWN])
        
        # Step 6: 构建结果
        result = self._build_result(
            posture_type, severity, strategy,
            angles, pressure_features, sensor_data
        )
        
        logger.debug(f"Posture classified: {posture_type.value} ({severity.value})")
        
        return result
    
    def _extract_angles(self, sensor_data: Dict) -> Dict[str, float]:
        """提取并规范化角度特征"""
        posture_angles = sensor_data.get('posture_angles', [0, 0, 0])
        
        return {
            'head': float(posture_angles[0]) if len(posture_angles) > 0 else 0.0,
            'shoulder': float(posture_angles[1]) if len(posture_angles) > 1 else 0.0,
            'pelvis': float(posture_angles[2]) if len(posture_angles) > 2 else 0.0,
        }
    
    def _analyze_pressure_matrix(self, sensor_data: Dict) -> PressureFeatures:
        """分析压力矩阵，计算派生特征"""
        pressure_matrix = np.array(sensor_data.get('pressure_matrix', [[0]*8 for _ in range(8)]))
        
        if pressure_matrix.shape != (8, 8):
            return PressureFeatures()
        
        # 归一化到 0-1
        max_val = np.max(pressure_matrix) if np.max(pressure_matrix) > 0 else 1
        pressure_norm = pressure_matrix / max_val
        
        # 左右分割（列 0-3 vs 4-7）
        left_half = pressure_norm[:, :4]
        right_half = pressure_norm[:, 4:]
        left_mean = np.mean(left_half)
        right_mean = np.mean(right_half)
        total = left_mean + right_mean
        
        left_right_balance = (right_mean - left_mean) / total if total > 0 else 0
        
        # 前后分割（行 0-3 vs 4-7）
        front_half = pressure_norm[:4, :]
        back_half = pressure_norm[4:, :]
        front_mean = np.mean(front_half)
        back_mean = np.mean(back_half)
        total_fb = front_mean + back_mean
        
        front_back_balance = (front_mean - back_mean) / total_fb if total_fb > 0 else 0
        
        # 不对称指数
        diff = np.abs(left_half - right_mean)
        asymmetry = np.mean(diff)
        
        # 压力重心
        rows, cols = np.meshgrid(np.arange(8), np.arange(8))
        weighted_rows = np.sum(pressure_norm * rows, axis=(0, 1))
        weighted_cols = np.sum(pressure_norm * cols, axis=(0, 1))
        total_weight = np.sum(pressure_norm)
        
        if total_weight > 0:
            cy = weighted_rows / (total_weight * 7)  # 归一化到 0-1
            cx = weighted_cols / (total_weight * 7)
        else:
            cy, cx = 0.5, 0.5
        
        return PressureFeatures(
            left_right_balance=round(float(left_right_balance), 3),
            front_back_balance=round(float(front_back_balance), 3),
            asymmetry_index=round(float(asymmetry), 3),
            center_of_pressure=(round(float(cx), 3), round(float(cy), 3)),
            max_pressure=round(float(max_val), 3),
            mean_pressure=round(float(np.mean(pressure_matrix)), 3),
            left_ratio=round(float(left_mean), 3),
            right_ratio=round(float(right_mean), 3),
            front_ratio=round(float(front_mean), 3),
            back_ratio=round(float(back_mean), 3)
        )
    
    def _match_rules(
        self,
        angles: Dict[str, float],
        pressure: PressureFeatures,
        history: Optional[List[Dict]]
    ) -> PostureType:
        """规则匹配引擎 - 优化版"""

        head = angles['head']
        shoulder = angles['shoulder']
        pelvis = angles['pelvis']

        # 规则 1: 前倾/探头（高优先级）
        # 条件：头部明显前倾 + 肩部前倾
        if head > self.angle_thresholds['head_normal'] and shoulder > self.angle_thresholds['shoulder_normal']:
            if abs(head) >= self.angle_thresholds['head_slight'] or abs(shoulder) >= self.angle_thresholds['shoulder_slight']:
                return PostureType.FORWARD_LEAN

        # 规则 2: 后仰/瘫坐（高优先级）
        if shoulder < -self.angle_thresholds['recline_threshold']:
            return PostureType.BACKWARD_RECLINE
        if pressure.front_back_balance < -0.30 and abs(head) < 8:
            return PostureType.BACKWARD_RECLINE

        # 规则 3: 侧偏（左/右）（高优先级）
        if abs(pelvis) > self.angle_thresholds['lateral_threshold']:
            return PostureType.LATERAL_TILT

        # 规则 4: 跷二郎腿（单侧严重承重）- 需要极端的单侧压力
        single_side_threshold = 0.60  # 单侧占比 > 60%
        if pressure.left_ratio > single_side_threshold or pressure.right_ratio > single_side_threshold:
            return PostureType.LEG_CROSSED

        # 规则 5: 交叉腿坐（中度不对称 + 无明显前后倾斜）
        # 必须同时满足：中度不对称 + 压力重心不在极端前部
        if (pressure.asymmetry_index > self.pressure_thresholds['asymmetry_mild'] and  # 降低到 mild
            pressure.front_back_balance < 0.30):  # 排除前伸坐姿
            return PostureType.CROSSED_LEGS

        # 规则 6: 盘腿坐（压力分散在两侧和后部）
        if (pressure.left_ratio > 0.40 and pressure.right_ratio > 0.40 and
            pressure.back_ratio > 0.35 and pressure.asymmetry_index < 0.15):
            return PostureType.LOTUS_POSITION

        # 规则 7: 前伸坐姿（重心明显前移 + 角度辅助验证）
        # 必须同时满足：压力重心明显前移 + 头部/肩部有明显前倾趋势
        if (pressure.front_back_balance > self.pressure_thresholds['front_reach_threshold'] + 0.05 and
            pressure.center_of_pressure[0] < 0.35 and
            (head > 15 or shoulder > 12)):
            return PostureType.FORWARD_REACH

        # 默认：正常坐姿
        return PostureType.NORMAL
    
    def _calculate_severity(
        self,
        angles: Dict[str, float],
        pressure: PressureFeatures,
        posture_type: PostureType
    ) -> SeverityLevel:
        """计算严重程度"""
        
        if posture_type == PostureType.NORMAL:
            max_deviation = max(abs(v) for v in angles.values())
            if max_deviation < self.angle_thresholds['pelvis_normal']:
                return SeverityLevel.IDEAL
            elif max_deviation < self.angle_thresholds['pelvis_slight']:
                return SeverityLevel.GOOD
            return SeverityLevel.GOOD
        
        # 计算综合偏差分数 (0-100)
        angle_score = 0
        angle_score += min(abs(angles['head']) / 45.0 * 35, 35)   # 头部权重 35%
        angle_score += min(abs(angles['shoulder']) / 30.0 * 30, 30)  # 肩部权重 30%
        angle_score += min(abs(angles['pelvis']) / 25.0 * 35, 35)   # 骨盆权重 35%
        
        pressure_score = pressure.asymmetry_index * 20  # 不对称权重 20%
        
        total_score = angle_score + pressure_score
        
        if total_score < 20:
            return SeverityLevel.GOOD
        elif total_score < 50:
            return SeverityLevel.WARNING
        else:
            return SeverityLevel.DANGER
    
    def _build_result(
        self,
        posture_type: PostureType,
        severity: SeverityLevel,
        strategy: Dict[str, Any],
        angles: Dict[str, float],
        pressure: PressureFeatures,
        sensor_data: Dict[str, Any]
    ) -> PostureResult:
        """构建完整的结果对象"""
        
        # 计算置信度（基于偏差程度的反比）
        max_deviation = max(abs(v) for v in angles.values()) if angles else 0
        base_confidence = 0.85
        deviation_penalty = min(max_deviation / 30.0 * 0.15, 0.15)
        confidence = max(base_confidence - deviation_penalty, 0.6)
        
        # 处理消息模板中的动态字段
        message = strategy.get('message', '')
        
        direction = "左" if angles.get('pelvis', 0) < 0 else "右" if abs(angles.get('pelvis', 0)) > 5 else ""
        opposite = "右" if direction == "左" else "左"
        side = "左" if pressure.left_ratio > pressure.right_ratio else "右"
        dominant = "左" if pressure.left_ratio > pressure.right_ratio else "右"

        try:
            message = message.format(
                direction=direction,
                opposite=opposite,
                side=side,
                dominant=dominant
            )
        except (KeyError, IndexError) as e:
            logger.debug(f"Message formatting error: {e}, using original message")

        # 处理 adjustments 中的动态键
        adjustments = dict(strategy.get('adjustments', {}))
        final_adjustments = {}
        for k, v in adjustments.items():
            try:
                if '{' not in k and '[' not in k:
                    final_adjustments[k] = v
                elif '[' in k:
                    eval_key = eval(k, {'__builtins__': {}}, {'left': 'left', 'right': 'right'})
                    final_adjustments[eval_key] = v
            except Exception as e:
                logger.debug(f"Skipping adjustment key {k}: {e}")

        # 处理 risk_areas 和 exercises 中的动态内容
        risk_areas = []
        for area in strategy.get('risk_areas', []):
            try:
                if '{' in area:
                    risk_areas.append(area.format(side=side, dominant=dominant))
                else:
                    risk_areas.append(area)
            except (KeyError, IndexError):
                risk_areas.append(area)

        exercises = []
        for ex in strategy.get('exercises', []):
            try:
                if '{' in ex:
                    exercises.append(ex.format(side=side))
                else:
                    exercises.append(ex)
            except (KeyError, IndexError):
                exercises.append(ex)

        return PostureResult(
            posture_type=posture_type,
            severity=strategy.get('severity', severity),
            confidence=round(confidence, 3),
            risk_areas=risk_areas,
            recommended_exercises=exercises,
            primary_adjustments={k: round(v, 4) for k, v in final_adjustments.items()},
            message=message,
            raw_angles=angles,
            pressure_features=pressure
        )
    
    def get_posture_name_cn(self, posture_type: PostureType) -> str:
        """获取姿态的中文名称"""
        names = {
            PostureType.NORMAL: "正常坐姿",
            PostureType.FORWARD_LEAN: "前倾/探头",
            PostureType.BACKWARD_RECLINE: "后仰/瘫坐",
            PostureType.LATERAL_TILT: "侧偏坐姿",
            PostureType.CROSSED_LEGS: "交叉腿坐",
            PostureType.LEG_CROSSED: "跷二郎腿",
            PostureType.LOTUS_POSITION: "盘腿坐",
            PostureType.FORWARD_REACH: "前伸坐姿",
            PostureType.UNKNOWN: "未知姿态",
        }
        return names.get(posture_type, "未知")
    
    @staticmethod
    def is_available() -> bool:
        """检查分类器可用性"""
        return True


def classify_posture(
    sensor_data: Dict[str, Any],
    history: Optional[List[Dict]] = None
) -> PostureResult:
    """
    便捷函数：快速分类姿态
    
    Args:
        sensor_data: 传感器数据
        history: 可选历史数据
        
    Returns:
        PostureResult: 分类结果
    """
    classifier = PostureClassifier()
    return classifier.classify(sensor_data, history)
