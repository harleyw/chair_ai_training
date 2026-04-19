#!/usr/bin/env python3

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
姿态分类器测试脚本
验证 8 种基本姿态类型的识别、边界情况、严重程度评估和 API 兼容性
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.posture_classifier import (
    PostureClassifier,
    PostureType,
    SeverityLevel,
    classify_posture
)


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_test_result(test_name: str, passed: bool, detail: str = ""):
    """打印测试结果"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} - {test_name}")
    if detail:
        print(f"         {detail}")


def test_basic_posture_types():
    """测试 1: 8 种基本姿态类型的正确识别"""
    print_separator("Test 1: 基本姿态类型识别")

    classifier = PostureClassifier()
    passed_count = 0
    total_tests = 8

    # 1. 正常坐姿 - 压力分布相对均匀，中心略偏后
    normal_data = {
        "posture_angles": [5.0, -3.0, 4.0],
        "pressure_matrix": [
            [0.08, 0.12, 0.20, 0.24, 0.23, 0.20, 0.13, 0.08],
            [0.10, 0.15, 0.24, 0.29, 0.27, 0.23, 0.15, 0.10],
            [0.07, 0.11, 0.19, 0.25, 0.23, 0.19, 0.12, 0.07],
            [0.05, 0.08, 0.14, 0.19, 0.17, 0.14, 0.09, 0.05],
            [0.03, 0.06, 0.11, 0.15, 0.14, 0.11, 0.07, 0.04],
            [0.02, 0.04, 0.09, 0.12, 0.11, 0.09, 0.05, 0.03],
            [0.02, 0.03, 0.06, 0.10, 0.09, 0.06, 0.04, 0.02],
            [0.01, 0.02, 0.04, 0.07, 0.06, 0.04, 0.02, 0.01]
        ]
    }
    result = classifier.classify(normal_data)
    is_pass = result.posture_type == PostureType.NORMAL
    passed_count += int(is_pass)
    print_test_result("正常坐姿 (normal)", is_pass,
                     f"Expected: normal, Got: {result.posture_type.value}")

    # 2. 前倾/探头
    forward_lean_data = {
        "posture_angles": [25.0, 18.0, 8.0],
        "pressure_matrix": [
            [0.15, 0.25, 0.40, 0.45, 0.35, 0.25, 0.15, 0.08],
            [0.18, 0.30, 0.48, 0.55, 0.42, 0.30, 0.18, 0.10],
            [0.12, 0.22, 0.38, 0.45, 0.35, 0.22, 0.12, 0.06],
            [0.08, 0.15, 0.28, 0.35, 0.26, 0.15, 0.08, 0.04],
            [0.05, 0.10, 0.20, 0.25, 0.18, 0.10, 0.05, 0.03],
            [0.03, 0.07, 0.15, 0.20, 0.14, 0.07, 0.03, 0.02],
            [0.02, 0.05, 0.10, 0.15, 0.10, 0.05, 0.02, 0.01],
            [0.01, 0.03, 0.07, 0.10, 0.07, 0.03, 0.01, 0.00]
        ]
    }
    result = classifier.classify(forward_lean_data)
    is_pass = result.posture_type == PostureType.FORWARD_LEAN
    passed_count += int(is_pass)
    print_test_result("前倾/探头 (forward_lean)", is_pass,
                     f"Expected: forward_lean, Got: {result.posture_type.value}")

    # 3. 后仰/瘫坐
    backward_recline_data = {
        "posture_angles": [-3.0, -25.0, -5.0],
        "pressure_matrix": [
            [0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.20, 0.12],
            [0.06, 0.10, 0.15, 0.20, 0.28, 0.32, 0.25, 0.15],
            [0.04, 0.07, 0.12, 0.16, 0.24, 0.28, 0.22, 0.13],
            [0.03, 0.05, 0.10, 0.13, 0.20, 0.24, 0.18, 0.10],
            [0.02, 0.04, 0.08, 0.11, 0.17, 0.20, 0.15, 0.08],
            [0.02, 0.03, 0.06, 0.09, 0.14, 0.17, 0.12, 0.06],
            [0.01, 0.02, 0.05, 0.07, 0.11, 0.14, 0.09, 0.04],
            [0.00, 0.01, 0.03, 0.05, 0.08, 0.11, 0.06, 0.03]
        ]
    }
    result = classifier.classify(backward_recline_data)
    is_pass = result.posture_type == PostureType.BACKWARD_RECLINE
    passed_count += int(is_pass)
    print_test_result("后仰/瘫坐 (backward_recline)", is_pass,
                     f"Expected: backward_recline, Got: {result.posture_type.value}")

    # 4. 左偏/右偏
    lateral_tilt_data = {
        "posture_angles": [5.0, -8.0, 18.0],
        "pressure_matrix": [
            [0.08, 0.12, 0.18, 0.22, 0.15, 0.10, 0.06, 0.03],
            [0.10, 0.15, 0.24, 0.30, 0.20, 0.13, 0.08, 0.04],
            [0.07, 0.11, 0.19, 0.25, 0.17, 0.11, 0.07, 0.03],
            [0.05, 0.08, 0.14, 0.19, 0.13, 0.08, 0.05, 0.02],
            [0.03, 0.06, 0.11, 0.15, 0.10, 0.06, 0.04, 0.02],
            [0.02, 0.04, 0.09, 0.12, 0.08, 0.05, 0.03, 0.01],
            [0.01, 0.03, 0.06, 0.09, 0.06, 0.04, 0.02, 0.01],
            [0.01, 0.02, 0.04, 0.06, 0.04, 0.03, 0.01, 0.00]
        ]
    }
    result = classifier.classify(lateral_tilt_data)
    is_pass = result.posture_type == PostureType.LATERAL_TILT
    passed_count += int(is_pass)
    print_test_result("左偏/右偏 (lateral_tilt)", is_pass,
                     f"Expected: lateral_tilt, Got: {result.posture_type.value}")

    # 5. 交叉腿坐 - 明显的左右不对称
    crossed_legs_data = {
        "posture_angles": [6.0, -5.0, 7.0],
        "pressure_matrix": [
            [0.15, 0.22, 0.32, 0.38, 0.25, 0.16, 0.09, 0.04],
            [0.18, 0.28, 0.42, 0.50, 0.32, 0.20, 0.11, 0.05],
            [0.14, 0.21, 0.33, 0.42, 0.27, 0.17, 0.10, 0.04],
            [0.09, 0.15, 0.25, 0.33, 0.21, 0.13, 0.07, 0.03],
            [0.06, 0.11, 0.19, 0.25, 0.16, 0.10, 0.05, 0.02],
            [0.04, 0.08, 0.15, 0.20, 0.12, 0.07, 0.04, 0.02],
            [0.03, 0.05, 0.11, 0.15, 0.09, 0.06, 0.03, 0.01],
            [0.02, 0.03, 0.07, 0.11, 0.07, 0.04, 0.02, 0.01]
        ]
    }
    result = classifier.classify(crossed_legs_data)
    is_pass = result.posture_type in [PostureType.CROSSED_LEGS, PostureType.LEG_CROSSED]
    passed_count += int(is_pass)
    print_test_result("交叉腿坐 (crossed_legs)", is_pass,
                     f"Expected: crossed_legs or leg_crossed, Got: {result.posture_type.value}")

    # 6. 跷二郎腿（单侧极端承重 > 60%）
    leg_crossed_data = {
        "posture_angles": [4.0, -6.0, 10.0],
        "pressure_matrix": [
            [0.03, 0.05, 0.08, 0.82, 0.62, 0.35, 0.14, 0.05],
            [0.04, 0.07, 0.11, 0.92, 0.70, 0.40, 0.17, 0.06],
            [0.03, 0.06, 0.09, 0.85, 0.64, 0.36, 0.15, 0.05],
            [0.02, 0.04, 0.07, 0.72, 0.52, 0.29, 0.12, 0.04],
            [0.01, 0.03, 0.05, 0.58, 0.42, 0.23, 0.09, 0.03],
            [0.01, 0.02, 0.04, 0.46, 0.34, 0.18, 0.07, 0.02],
            [0.01, 0.02, 0.03, 0.35, 0.26, 0.14, 0.06, 0.02],
            [0.00, 0.01, 0.02, 0.26, 0.19, 0.10, 0.04, 0.01]
        ]
    }
    result = classifier.classify(leg_crossed_data)
    is_pass = result.posture_type == PostureType.LEG_CROSSED
    passed_count += int(is_pass)
    print_test_result("跷二郎腿 (leg_crossed)", is_pass,
                     f"Expected: leg_crossed, Got: {result.posture_type.value}")

    # 7. 盘腿坐（压力分散在两侧和后部，对称性好）
    lotus_position_data = {
        "posture_angles": [3.0, -4.0, 5.0],
        "pressure_matrix": [
            [0.14, 0.20, 0.23, 0.20, 0.18, 0.20, 0.23, 0.17],
            [0.17, 0.25, 0.30, 0.26, 0.23, 0.26, 0.28, 0.21],
            [0.13, 0.19, 0.25, 0.22, 0.20, 0.22, 0.24, 0.18],
            [0.09, 0.15, 0.20, 0.18, 0.16, 0.18, 0.19, 0.13],
            [0.07, 0.11, 0.16, 0.14, 0.13, 0.14, 0.15, 0.10],
            [0.05, 0.09, 0.13, 0.11, 0.10, 0.11, 0.12, 0.08],
            [0.04, 0.06, 0.10, 0.09, 0.08, 0.09, 0.10, 0.06],
            [0.02, 0.04, 0.07, 0.06, 0.05, 0.06, 0.07, 0.04]
        ]
    }
    result = classifier.classify(lotus_position_data)
    is_pass = result.posture_type == PostureType.LOTUS_POSITION
    passed_count += int(is_pass)
    print_test_result("盘腿坐 (lotus_position)", is_pass,
                     f"Expected: lotus_position, Got: {result.posture_type.value}")

    # 8. 前伸坐姿（重心明显前移 + 明显角度前倾 + 左右对称）
    forward_reach_data = {
        "posture_angles": [18.0, 14.0, 6.0],
        "pressure_matrix": [
            [0.28, 0.42, 0.55, 0.58, 0.32, 0.18, 0.09, 0.04],
            [0.32, 0.48, 0.65, 0.70, 0.38, 0.22, 0.11, 0.05],
            [0.26, 0.40, 0.52, 0.57, 0.31, 0.18, 0.09, 0.04],
            [0.19, 0.29, 0.40, 0.44, 0.24, 0.14, 0.07, 0.03],
            [0.13, 0.21, 0.30, 0.33, 0.18, 0.10, 0.05, 0.02],
            [0.09, 0.15, 0.22, 0.25, 0.14, 0.08, 0.04, 0.02],
            [0.06, 0.10, 0.16, 0.18, 0.10, 0.06, 0.03, 0.01],
            [0.04, 0.07, 0.11, 0.13, 0.07, 0.04, 0.02, 0.01]
        ]
    }
    result = classifier.classify(forward_reach_data)
    is_pass = result.posture_type == PostureType.FORWARD_REACH
    passed_count += int(is_pass)
    print_test_result("前伸坐姿 (forward_reach)", is_pass,
                     f"Expected: forward_reach, Got: {result.posture_type.value}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def test_edge_cases():
    """测试 2: 边界情况和多特征冲突时的优先级"""
    print_separator("Test 2: 边界情况和优先级")

    classifier = PostureClassifier()
    passed_count = 0
    total_tests = 5

    # 边界情况 1: 多特征冲突（前倾 + 后仰特征同时存在）
    conflict_data = {
        "posture_angles": [30.0, -25.0, 15.0],
        "pressure_matrix": [[0.1] * 8 for _ in range(8)]
    }
    result = classifier.classify(conflict_data)
    is_pass = result.posture_type in [PostureType.FORWARD_LEAN, PostureType.BACKWARD_RECLINE, PostureType.LATERAL_TILT]
    passed_count += int(is_pass)
    print_test_result("多特征冲突时能正确分类", is_pass,
                     f"Got: {result.posture_type.value} (应优先匹配最显著特征)")

    # 边界情况 2: 极端角度值
    extreme_angles = {
        "posture_angles": [50.0, -45.0, 35.0],
        "pressure_matrix": [[0.2] * 8 for _ in range(8)]
    }
    result = classifier.classify(extreme_angles)
    is_pass = result.severity == SeverityLevel.DANGER
    passed_count += int(is_pass)
    print_test_result("极端角度值触发 DANGER 级别", is_pass,
                     f"Severity: {result.severity.value}")

    # 边界情况 3: 全零输入
    zero_input = {
        "posture_angles": [0.0, 0.0, 0.0],
        "pressure_matrix": [[0.0] * 8 for _ in range(8)]
    }
    result = classifier.classify(zero_input)
    is_pass = result.posture_type == PostureType.NORMAL and result.confidence > 0
    passed_count += int(is_pass)
    print_test_result("全零输入返回正常坐姿", is_pass,
                     f"Posture: {result.posture_type.value}, Confidence: {result.confidence}")

    # 边界情况 4: 缺失字段处理
    missing_fields = {"posture_angles": [10, -5, 8]}
    try:
        result = classifier.classify(missing_fields)
        is_pass = True
        passed_count += int(is_pass)
        print_test_result("缺失压力矩阵时不崩溃", is_pass,
                        "使用默认值继续运行")
    except Exception as e:
        is_pass = False
        passed_count += int(is_pass)
        print_test_result("缺失压力矩阵时不崩溃", is_pass,
                        f"Error: {e}")

    # 边界情况 5: 历史数据影响
    history_data = [
        {"posture_angles": [5, -3, 4], "pressure_matrix": [[0.1]*8 for _ in range(8)]},
        {"posture_angles": [8, -5, 6], "pressure_matrix": [[0.12]*8 for _ in range(8)]},
        {"posture_angles": [12, -8, 9], "pressure_matrix": [[0.15]*8 for _ in range(8)]}
    ]
    current_data = {
        "posture_angles": [20, -12, 14],
        "pressure_matrix": [[0.18]*8 for _ in range(8)]
    }
    result_with_history = classifier.classify(current_data, history=history_data)
    result_without_history = classifier.classify(current_data)
    
    is_pass = result_with_history.posture_type == result_without_history.posture_type
    passed_count += int(is_pass)
    print_test_result("历史数据不改变基础分类结果", is_pass,
                    f"With history: {result_with_history.posture_type.value}, "
                    f"Without: {result_without_history.posture_type.value}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def test_severity_levels():
    """测试 3: 严重程度分级的准确性"""
    print_separator("Test 3: 严重程度分级")

    classifier = PostureClassifier()
    passed_count = 0
    total_tests = 5

    # 理想状态
    ideal_data = {
        "posture_angles": [2.0, -1.0, 3.0],
        "pressure_matrix": [[0.1] * 8 for _ in range(8)]
    }
    result = classifier.classify(ideal_data)
    is_pass = result.severity == SeverityLevel.IDEAL or result.severity == SeverityLevel.GOOD
    passed_count += int(is_pass)
    print_test_result("理想状态 → IDEAL/GOOD", is_pass,
                     f"Got: {result.severity.value}")

    # 轻微偏差
    slight_deviation = {
        "posture_angles": [12.0, -8.0, 10.0],
        "pressure_matrix": [[0.15] * 8 for _ in range(8)]
    }
    result = classifier.classify(slight_deviation)
    is_pass = result.severity in [SeverityLevel.GOOD, SeverityLevel.WARNING]
    passed_count += int(is_pass)
    print_test_result("轻微偏差 → GOOD/WARNING", is_pass,
                     f"Got: {result.severity.value}")

    # 中等偏差
    moderate_deviation = {
        "posture_angles": [22.0, -16.0, 18.0],
        "pressure_matrix": [[0.2] * 8 for _ in range(8)]
    }
    result = classifier.classify(moderate_deviation)
    is_pass = result.severity == SeverityLevel.WARNING
    passed_count += int(is_pass)
    print_test_result("中等偏差 → WARNING", is_pass,
                     f"Got: {result.severity.value}")

    # 严重偏差
    severe_deviation = {
        "posture_angles": [40.0, -30.0, 28.0],
        "pressure_matrix": [[0.3] * 8 for _ in range(8)]
    }
    result = classifier.classify(severe_deviation)
    is_pass = result.severity == SeverityLevel.DANGER
    passed_count += int(is_pass)
    print_test_result("严重偏差 → DANGER", is_pass,
                     f"Got: {result.severity.value}")

    # 跷二郎腿应为 DANGER
    leg_crossed = {
        "posture_angles": [5, -5, 8],
        "pressure_matrix": [
            [0.05, 0.08, 0.12, 0.80, 0.60, 0.32, 0.13, 0.05],
            [0.07, 0.11, 0.17, 0.90, 0.68, 0.37, 0.16, 0.06],
            [0.05, 0.09, 0.14, 0.82, 0.61, 0.33, 0.14, 0.05],
            [0.03, 0.06, 0.10, 0.70, 0.50, 0.27, 0.11, 0.04],
            [0.02, 0.04, 0.08, 0.56, 0.40, 0.22, 0.09, 0.03],
            [0.02, 0.03, 0.06, 0.45, 0.32, 0.17, 0.07, 0.02],
            [0.01, 0.02, 0.04, 0.34, 0.24, 0.13, 0.05, 0.02],
            [0.00, 0.01, 0.03, 0.26, 0.18, 0.10, 0.04, 0.01]
        ]
    }
    result = classifier.classify(leg_crossed)
    is_pass = result.severity == SeverityLevel.DANGER
    passed_count += int(is_pass)
    print_test_result("跷二郎腿 → DANGER", is_pass,
                     f"Got: {result.severity.value}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def test_adjustment_strategies():
    """测试 4: 建议策略的完整性"""
    print_separator("Test 4: 建议策略完整性")

    classifier = PostureClassifier()
    passed_count = 0
    total_tests = 8

    posture_types_to_test = [
        (PostureType.NORMAL, "正常坐姿"),
        (PostureType.FORWARD_LEAN, "前倾/探头"),
        (PostureType.BACKWARD_RECLINE, "后仰/瘫坐"),
        (PostureType.LATERAL_TILT, "侧偏坐姿"),
        (PostureType.CROSSED_LEGS, "交叉腿坐"),
        (PostureType.LEG_CROSSED, "跷二郎腿"),
        (PostureType.LOTUS_POSITION, "盘腿坐"),
        (PostureType.FORWARD_REACH, "前伸坐姿")
    ]

    for posture_type, name_cn in posture_types_to_test:
        if posture_type == PostureType.NORMAL:
            data = {"posture_angles": [2, -1, 3], "pressure_matrix": [[0.1]*8 for _ in range(8)]}
        elif posture_type == PostureType.FORWARD_LEAN:
            data = {"posture_angles": [25, 18, 8], "pressure_matrix": [[0.2]*8 for _ in range(8)]}
        elif posture_type == PostureType.BACKWARD_RECLINE:
            data = {"posture_angles": [-3, -25, -5], "pressure_matrix": [[0.15]*8 for _ in range(8)]}
        elif posture_type == PostureType.LATERAL_TILT:
            data = {"posture_angles": [5, -8, 18], "pressure_matrix": [[0.12]*8 for _ in range(8)]}
        elif posture_type == PostureType.CROSSED_LEGS:
            data = {"posture_angles": [6, -5, 7], "pressure_matrix": [[0.18]*8 for _ in range(8)]}
        elif posture_type == PostureType.LEG_CROSSED:
            data = {"posture_angles": [4, -6, 10], "pressure_matrix": [[0.05, 0.08, 0.12, 0.80, 0.55, 0.30, 0.12, 0.04] for _ in range(8)]}
        elif posture_type == PostureType.LOTUS_POSITION:
            data = {"posture_angles": [3, -4, 5], "pressure_matrix": [[0.15, 0.22, 0.25, 0.22, 0.20, 0.22, 0.25, 0.18] for _ in range(8)]}
        else:
            data = {"posture_angles": [12, 8, 5], "pressure_matrix": [[0.25, 0.38, 0.52, 0.58, 0.35, 0.20, 0.10, 0.04] for _ in range(8)]}

        result = classifier.classify(data)

        has_message = bool(result.message)
        has_risk_areas = len(result.risk_areas) > 0 or posture_type == PostureType.NORMAL
        has_exercises = len(result.recommended_exercises) > 0 or posture_type == PostureType.NORMAL
        has_adjustments = len(result.primary_adjustments) > 0 or posture_type == PostureType.NORMAL
        confidence_valid = 0 <= result.confidence <= 1

        is_pass = all([has_message, has_risk_areas, has_exercises, has_adjustments, confidence_valid])
        passed_count += int(is_pass)

        missing = []
        if not has_message: missing.append("message")
        if not has_risk_areas and posture_type != PostureType.NORMAL: missing.append("risk_areas")
        if not has_exercises and posture_type != PostureType.NORMAL: missing.append("exercises")
        if not has_adjustments and posture_type != PostureType.NORMAL: missing.append("adjustments")
        if not confidence_valid: missing.append("confidence")

        print_test_result(f"{name_cn} ({posture_type.value})", is_pass,
                         f"Missing: {', '.join(missing) if missing else 'None'} | "
                         f"Confidence: {result.confidence:.3f}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def test_api_compatibility():
    """测试 5: 与现有 API 的兼容性"""
    print_separator("Test 5: API 兼容性测试")

    from api.service import ChairAIService
    from api.models import SensorData

    service = ChairAIService()
    passed_count = 0
    total_tests = 4

    # 测试 1: analyze_posture 返回格式向后兼容
    sensor_data_dict = {
        "pressure_matrix": [[0.1] * 8 for _ in range(8)],
        "posture_angles": [15.0, -8.0, 10.0],
        "sitting_duration": 1800.0,
        "user_weight": 70.0,
        "user_height": 1.70,
        "fatigue_level": 0.4
    }

    try:
        analysis, issues, comfort_score, risk_level = service.analyze_posture(sensor_data_dict)

        has_basic_fields = all([
            "head_posture" in analysis,
            "shoulder_posture" in analysis,
            "pelvis_posture" in analysis,
            "comfort_score" in analysis,
            "overall_risk_level" in analysis
        ])

        has_new_field = "posture_detail" in analysis

        is_pass = has_basic_fields and has_new_field
        passed_count += int(is_pass)
        print_test_result("analyze_posture 包含新旧字段", is_pass,
                         f"Basic fields: {has_basic_fields}, New field (posture_detail): {has_new_field}")
    except Exception as e:
        is_pass = False
        passed_count += int(is_pass)
        print_test_result("analyze_posture 包含新旧字段", is_pass,
                         f"Error: {e}")

    # 测试 2: posture_detail 结构正确
    try:
        analysis, _, _, _ = service.analyze_posture(sensor_data_dict)
        posture_detail = analysis.get("posture_detail")

        if posture_detail and isinstance(posture_detail, dict):
            has_required_keys = all([
                "posture_type" in posture_detail,
                "posture_name_cn" in posture_detail,
                "severity" in posture_detail,
                "confidence" in posture_detail,
                "message" in posture_detail
            ])
            is_pass = has_required_keys
            passed_count += int(is_pass)
            print_test_result("posture_detail 结构完整", is_pass,
                            f"Keys present: {list(posture_detail.keys())}")
        else:
            is_pass = posture_detail is None
            passed_count += int(is_pass)
            print_test_result("posture_detail 可为 None", is_pass,
                            "当分类器出错时应优雅降级")
    except Exception as e:
        is_pass = False
        passed_count += int(is_pass)
        print_test_result("posture_detail 结构完整", is_pass,
                         f"Error: {e}")

    # 测试 3: 使用 SensorData 模型调用
    try:
        sensor_data = SensorData(
            pressure_matrix=[[0.15] * 8 for _ in range(8)],
            posture_angles=[20.0, -12.0, 14.0],
            sitting_duration=3600.0,
            user_weight=72.0,
            user_height=1.68,
            fatigue_level=0.6
        )

        data_dict = sensor_data.model_dump()
        action, confidence = service.predict_action(data_dict)
        analysis, issues, comfort_score, risk_level = service.analyze_posture(data_dict)

        is_pass = len(action) == 8 and 0 <= confidence <= 1
        passed_count += int(is_pass)
        print_test_result("SensorData 模型集成正常", is_pass,
                         f"Action dim: {len(action)}, Confidence: {confidence:.3f}")
    except Exception as e:
        is_pass = False
        passed_count += int(is_pass)
        print_test_result("SensorData 模型集成正常", is_pass,
                         f"Error: {e}")

    # 测试 4: 无模型加载时的回退行为
    try:
        no_model_service = ChairAIService(model_path=None)
        
        test_data = {
            "pressure_matrix": [[0.2] * 8 for _ in range(8)],
            "posture_angles": [18, -10, 12],
            "sitting_duration": 2400,
            "user_weight": 68,
            "user_height": 1.65,
            "fatigue_level": 0.5
        }

        action, confidence = no_model_service.predict_action(test_data)
        analysis, issues, comfort_score, risk_level = no_model_service.analyze_posture(test_data)

        is_pass = len(action) == 8 and "posture_detail" in analysis
        passed_count += int(is_pass)
        print_test_result("无模型时姿态分类仍可用", is_pass,
                         f"Action available: {len(action)==8}, "
                         f"Posture detail available: {'posture_detail' in analysis}")
    except Exception as e:
        is_pass = False
        passed_count += int(is_pass)
        print_test_result("无模型时姿态分类仍可用", is_pass,
                         f"Error: {e}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def main():
    """主测试函数"""
    print("\n" + "🪑" * 35)
    print("   人体工学座椅 AI 系统 - 姿态分类器测试套件")
    print("   Posture Classifier Test Suite v2.0")
    print("🪑" * 35)

    results = []

    results.append(("基本姿态类型识别", test_basic_posture_types()))
    results.append(("边界情况和优先级", test_edge_cases()))
    results.append(("严重程度分级", test_severity_levels()))
    results.append(("建议策略完整性", test_adjustment_strategies()))
    results.append(("API 兼容性", test_api_compatibility()))

    print("\n" + "=" * 70)
    print("  📋 测试总结")
    print("=" * 70)

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")

    print("\n" + "-" * 70)
    print(f"  总计: {total_passed}/{total_tests} 测试组通过")

    if total_passed == total_tests:
        print("  🎉 所有测试通过！姿态分类器功能正常。")
        return 0
    else:
        print(f"  ⚠️  有 {total_tests - total_passed} 个测试组未通过，请检查上述失败项。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
