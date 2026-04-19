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
奖励函数配置系统测试套件
验证数据模型、API 端点、构建器、预览功能等的完整性和正确性
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime


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


def test_data_models():
    """测试 1: 数据模型验证"""
    print_separator("Test 1: 数据模型验证")

    from api.reward_config import (
        RewardConfig,
        RewardWeights,
        ComfortConfig,
        ThresholdConfig,
        AdvancedConfig,
        get_default_config,
        validate_config,
    )

    passed_count = 0
    total_tests = 8

    # 1.1 默认配置创建
    try:
        default = get_default_config()
        is_pass = default is not None and isinstance(default, RewardConfig)
        passed_count += int(is_pass)
        print_test_result("默认配置创建", is_pass, f"Type: {type(default).__name__}")
    except Exception as e:
        print_test_result("默认配置创建", False, f"Error: {e}")

    # 1.2 合法配置验证通过
    try:
        config = RewardConfig(
            config_name="test",
            weights=RewardWeights(comfort=1.0, pressure=0.8, static_penalty=0.5, energy=0.3),
            comfort=ComfortConfig(spine_alignment_weight=0.5, pressure_uniformity_weight=0.5),
            thresholds=ThresholdConfig(max_pressure=50.0, static_duration=900.0)
        )
        validation = validate_config(config.model_dump())
        is_pass = validation.valid and len(validation.errors) == 0
        passed_count += int(is_pass)
        print_test_result("合法配置验证通过", is_pass, f"Score: {validation.score:.1f}")
    except Exception as e:
        print_test_result("合法配置验证通过", False, f"Error: {e}")

    # 1.3 权重范围检查 - 负数拒绝
    try:
        invalid_weights = RewardWeights(comfort=-0.5, pressure=0.8, static_penalty=0.5, energy=0.3)
        print_test_result("负数权重被拒绝", False, "应该抛出 ValidationError")
    except Exception as e:
        is_pass = "weight" in str(e).lower() or "negative" in str(e).lower() or "validat" in str(e).lower()
        passed_count += int(is_pass)
        print_test_result("负数权重被拒绝", True, f"Correctly rejected: {str(e)[:60]}...")

    # 1.4 舒适度子项归一化检查
    try:
        invalid_comfort = ComfortConfig(spine_alignment_weight=0.8, pressure_uniformity_weight=0.3)
        print_test_result("非归一化舒适度被拒绝", False, "应该抛出 ValidationError")
    except Exception as e:
        is_pass = "normalization" in str(e).lower() or "1.0" in str(e) or "validat" in str(e).lower()
        passed_count += int(is_pass)
        print_test_result("非归一化舒适度被拒绝", True)

    # 1.5 阈值正数检查
    try:
        invalid_threshold = ThresholdConfig(max_pressure=-10.0, static_duration=900.0)
        print_test_result("负数阈值被拒绝", False, "应该抛出 ValidationError")
    except Exception as e:
        is_pass = True
        passed_count += int(is_pass)
        print_test_result("负数阈值被拒绝", True)

    # 1.6 高级选项 (可选字段)
    try:
        config_with_advanced = RewardConfig(
            advanced=AdvancedConfig(
                enable_fatigue_awareness=True,
                fatigue_penalty_weight=0.2,
                custom_formula="comfort * 0.9 - pressure_penalty * 1.1"
            )
        )
        is_pass = config_with_advanced.advanced is not None
        passed_count += int(is_pass)
        print_test_result("高级选项支持", is_pass, "Advanced config accepted")
    except Exception as e:
        print_test_result("高级选项支持", False, f"Error: {e}")

    # 1.7 自定义公式安全检查 - 危险代码拒绝
    try:
        dangerous = AdvancedConfig(custom_formula="__import__('os').system('ls')")
        print_test_result("危险公式被拒绝", False, "应该拒绝包含 __import__ 的公式")
    except Exception as e:
        is_pass = "unsafe" in str(e).lower() or "dangerous" in str(e).lower() or "not allowed" in str(e).lower() or "validat" in str(e).lower()
        passed_count += int(is_pass)
        print_test_result("危险公式被拒绝", True)

    # 1.8 配置评分系统
    try:
        good_config = get_default_config().model_dump()
        validation = validate_config(good_config)
        is_pass = 80 <= validation.score <= 100
        passed_count += int(is_pass)
        print_test_result("默认配置评分合理", is_pass, f"Score: {validation.score:.1f}")
    except Exception as e:
        print_test_result("默认配置评分合理", False, f"Error: {e}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def test_preset_management():
    """测试 2: 预设管理"""
    print_separator("Test 2: 预设管理")

    from api.reward_config import (
        get_preset,
        list_presets,
        BUILTIN_PRESETS,
        RewardConfig,
        RewardWeights,
        ComfortConfig,
        ThresholdConfig,
    )

    passed_count = 0
    total_tests = 6

    # 2.1 内置预设数量
    presets = list_presets()
    is_pass = len(presets) >= 5
    passed_count += int(is_pass)
    print_test_result("内置预设数量 >= 5", is_pass, f"Count: {len(presets)}")

    # 2.2 获取 balanced 预设
    balanced = get_preset("balanced")
    is_pass = balanced is not None and isinstance(balanced, RewardConfig)
    passed_count += int(is_pass)
    print_test_result("获取 balanced 预设", is_pass, f"Name: {balanced.config_name if balanced else 'None'}")

    # 2.3 获取 health_first 预设
    health = get_preset("health_first")
    is_pass = health is not None and health.weights.pressure > 1.0
    passed_count += int(is_pass)
    print_test_result("health_first 压力权重 > 1.0", is_pass, f"Pressure weight: {health.weights.pressure if health else 'N/A'}")

    # 2.4 获取不存在的预设
    nonexistent = get_preset("nonexistent_preset_xyz")
    is_pass = nonexistent is None
    passed_count += int(is_pass)
    print_test_result("不存在预设返回 None", is_pass)

    # 2.5 各预设配置名称唯一
    names = [p.name for p in presets]
    is_pass = len(names) == len(set(names))
    passed_count += int(is_pass)
    print_test_result("预设名称唯一性", is_pass)

    # 2.6 预设分类覆盖
    categories = set(p.category for p in presets)
    has_general = "general" in categories
    passed_count += int(has_general)
    print_test_result("包含 general 分类", has_general, f"Categories: {categories}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def test_reward_calculation():
    """测试 3: 奖励值计算"""
    print_separator("Test 3: 奖励值计算与分解")

    from api.reward_config import (
        RewardConfig,
        calculate_reward_breakdown,
        get_default_config,
        get_preset,
    )

    passed_count = 0
    total_tests = 7

    config = get_default_config()

    # 3.1 正常情况计算
    try:
        breakdown = calculate_reward_breakdown(
            config=config,
            action_magnitude=0.3,
            spine_curvature=0.15,
            pressure_variance=0.1,
            max_pressure=35.0,
            static_duration=300.0,
            fatigue_level=0.2
        )
        is_pass = breakdown.total_reward is not None
        passed_count += int(is_pass)
        print_test_result("正常情况计算成功", is_pass, f"Total: {breakdown.total_reward:.4f}")
    except Exception as e:
        print_test_result("正常情况计算成功", False, f"Error: {e}")

    # 3.2 分解项完整性
    try:
        breakdown = calculate_reward_breakdown(
            config=config,
            action_magnitude=0.3,
            spine_curvature=0.2,
            pressure_variance=0.15,
            max_pressure=40.0,
            static_duration=500.0
        )
        has_all_fields = all([
            hasattr(breakdown, 'total_reward'),
            hasattr(breakdown, 'comfort_component'),
            hasattr(breakdown, 'pressure_penalty'),
            hasattr(breakdown, 'static_penalty'),
            hasattr(breakdown, 'energy_penalty')
        ])
        is_pass = has_all_fields
        passed_count += int(is_pass)
        print_test_result("分解项完整性", is_pass)
    except Exception as e:
        print_test_result("分解项完整性", False, f"Error: {e}")

    # 3.3 高压力惩罚
    try:
        normal = calculate_reward_breakdown(config=config, max_pressure=30.0, action_magnitude=0.2, spine_curvature=0.1, pressure_variance=0.08, static_duration=200.0)
        high_pressure = calculate_reward_breakdown(config=config, max_pressure=80.0, action_magnitude=0.2, spine_curvature=0.1, pressure_variance=0.08, static_duration=200.0)
        is_pass = high_pressure.total_reward < normal.total_reward
        passed_count += int(is_pass)
        print_test_result("高压力导致奖励降低", is_pass, f"Normal: {normal.total_reward:.4f}, High P: {high_pressure.total_reward:.4f}")
    except Exception as e:
        print_test_result("高压力导致奖励降低", False, f"Error: {e}")

    # 3.4 静态时间惩罚
    try:
        short_time = calculate_reward_breakdown(config=config, static_duration=300.0, action_magnitude=0.2, spine_curvature=0.1, pressure_variance=0.08, max_pressure=35.0)
        long_time = calculate_reward_breakdown(config=config, static_duration=1800.0, action_magnitude=0.2, spine_curvature=0.1, pressure_variance=0.08, max_pressure=35.0)
        is_pass = long_time.total_reward < short_time.total_reward
        passed_count += int(is_pass)
        print_test_result("长时间静态惩罚生效", is_pass, f"Short: {short_time.total_reward:.4f}, Long: {long_time.total_reward:.4f}")
    except Exception as e:
        print_test_result("长时间静态惩罚生效", False, f"Error: {e}")

    # 3.5 大动作能量惩罚
    try:
        small_action = calculate_reward_breakdown(config=config, action_magnitude=0.1, spine_curvature=0.1, pressure_variance=0.08, max_pressure=35.0, static_duration=300.0)
        large_action = calculate_reward_breakdown(config=config, action_magnitude=0.8, spine_curvature=0.1, pressure_variance=0.08, max_pressure=35.0, static_duration=300.0)
        is_pass = large_action.total_reward < small_action.total_reward
        passed_count += int(is_pass)
        print_test_result("大动作幅度惩罚生效", is_pass, f"Small: {small_action.total_reward:.4f}, Large: {large_action.total_reward:.4f}")
    except Exception as e:
        print_test_result("大动作幅度惩罚生效", False, f"Error: {e}")

    # 3.6 不同预设产生不同结果
    try:
        balanced_cfg = get_preset("balanced")
        strict_cfg = get_preset("strict_posture")

        same_input = dict(action_magnitude=0.3, spine_curvature=0.25, pressure_variance=0.18, max_pressure=45.0, static_duration=600.0)

        r_balanced = calculate_reward_breakdown(config=balanced_cfg, **same_input)
        r_strict = calculate_reward_breakdown(config=strict_cfg, **same_input)

        is_pass = r_balanced.total_reward != r_strict.total_reward
        passed_count += int(is_pass)
        print_test_result("不同预设产生不同结果", is_pass, f"Balanced: {r_balanced.total_reward:.4f}, Strict: {r_strict.total_reward:.4f}")
    except Exception as e:
        print_test_result("不同预设产生不同结果", False, f"Error: {e}")

    # 3.7 疲劳感知 (如果启用)
    try:
        no_fatigue = calculate_reward_breakdown(config=config, fatigue_level=0.0, posture_changed=False, symmetry_ratio=1.0, action_magnitude=0.2, spine_curvature=0.1, pressure_variance=0.08, max_pressure=35.0, static_duration=400.0)
        with_fatigue = calculate_reward_breakdown(config=config, fatigue_level=0.8, posture_changed=False, symmetry_ratio=1.0, action_magnitude=0.2, spine_curvature=0.1, pressure_variance=0.08, max_pressure=35.0, static_duration=400.0)

        if with_fatigue.fatigue_penalty is not None:
            is_pass = with_fatigue.total_reward < no_fatigue.total_reward
            passed_count += int(is_pass)
            print_test_result("疲劳惩罚生效", is_pass, f"No fatigue: {no_fatigue.total_reward:.4f}, Fatigue: {with_fatigue.total_reward:.4f}")
        else:
            is_pass = with_fatigue.fatigue_penalty is None and config.advanced is None or not config.advanced.enable_fatigue_awareness
            passed_count += int(is_pass)
            print_test_result("疲劳惩罚正确禁用", is_pass)
    except Exception as e:
        print_test_result("疲劳惩罚测试", False, f"Error: {e}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def test_dynamic_builder():
    """测试 4: 动态构建器"""
    print_separator("Test 4: 动态奖励函数构建器")

    from training.dynamic_rewards import (
        RewardFunctionBuilder,
        DynamicRewardFunction,
        HotReloadableEnv,
    )
    from api.reward_config import get_default_config, get_preset

    passed_count = 0
    total_tests = 7

    builder = RewardFunctionBuilder()

    # 4.1 构建默认函数
    try:
        default_fn = builder.build_default()
        is_pass = isinstance(default_fn, DynamicRewardFunction)
        passed_count += int(is_pass)
        print_test_result("构建默认奖励函数", is_pass, f"Version: {default_fn.version}")
    except Exception as e:
        print_test_result("构建默认奖励函数", False, f"Error: {e}")

    # 4.2 从预设构建
    try:
        balanced_fn = builder.build_from_preset("balanced")
        is_pass = isinstance(balanced_fn, DynamicRewardFunction) and balanced_fn.version > 0
        passed_count += int(is_pass)
        print_test_result("从预设构建函数", is_pass, f"Preset: balanced, Version: {balanced_fn.version}")
    except Exception as e:
        print_test_result("从预设构建函数", False, f"Error: {e}")

    # 4.3 从字典构建
    try:
        config_dict = get_default_config().model_dump()
        dict_fn = builder.build_from_config(config_dict)
        is_pass = isinstance(dict_fn, DynamicRewardFunction)
        passed_count += int(is_pass)
        print_test_result("从字典构建函数", is_pass)
    except Exception as e:
        print_test_result("从字典构建函数", False, f"Error: {e}")

    # 4.4 函数可调用
    try:
        fn = builder.build_default()
        result = fn(
            action_magnitude=0.3,
            spine_curvature=0.15,
            pressure_variance=0.12,
            max_pressure=38.0,
            static_duration=450.0
        )
        is_pass = result is not None and hasattr(result, 'total_reward') and hasattr(result, 'breakdown')
        passed_count += int(is_pass)
        print_test_result("函数可正常调用", is_pass, f"Result type: {type(result).__name__}")
    except Exception as e:
        print_test_result("函数可正常调用", False, f"Error: {e}")

    # 4.5 调用计数器
    try:
        fn = builder.build_default()
        for _ in range(5):
            fn(action_magnitude=0.2, spine_curvature=0.1, pressure_variance=0.08, max_pressure=32.0, static_duration=200.0)
        is_pass = fn.call_count == 5
        passed_count += int(is_pass)
        print_test_result("调用计数器工作", is_pass, f"Count: {fn.call_count}")
    except Exception as e:
        print_test_result("调用计数器工作", False, f"Error: {e}")

    # 4.6 版本号递增
    try:
        fn1 = builder.build_default()
        fn2 = builder.build_default()
        is_pass = fn2.version > fn1.version
        passed_count += int(is_pass)
        print_test_result("版本号自动递增", is_pass, f"V1: {fn1.version}, V2: {fn2.version}")
    except Exception as e:
        print_test_result("版本号自动递增", False, f"Error: {e}")

    # 4.7 单例模式
    try:
        builder2 = RewardFunctionBuilder()
        is_pass = builder is builder2
        passed_count += int(is_pass)
        print_test_result("构建器单例模式", is_pass)
    except Exception as e:
        print_test_result("构建器单例模式", False, f"Error: {e}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def test_hot_reload():
    """测试 5: 热更新机制"""
    print_separator("Test 5: 热更新机制")

    from training.dynamic_rewards import (
        HotReloadableEnv,
        RewardFunctionBuilder,
    )
    from api.reward_config import (
        RewardConfig,
        RewardWeights,
        ComfortConfig,
        ThresholdConfig,
        AdvancedConfig,
        get_default_config,
    )

    passed_count = 0
    total_tests = 6

    class MockEnv:
        def __init__(self):
            self.reset_called = False
        
        def reset(self):
            self.reset_called = True
    
    # 5.1 创建热更新环境
    try:
        mock_env = MockEnv()
        hot_env = HotReloadableEnv(mock_env)
        is_pass = hot_env.current_function is not None
        passed_count += int(is_pass)
        print_test_result("创建热更新环境", is_pass)
    except Exception as e:
        print_test_result("创建热更新环境", False, f"Error: {e}")

    # 5.2 更新配置
    try:
        hot_env = HotReloadableEnv(MockEnv())
        
        new_config = RewardConfig(
            config_name="test_update",
            weights=RewardWeights(comfort=1.5, pressure=1.2, static_penalty=0.8, energy=0.5),
            comfort=ComfortConfig(),
            thresholds=ThresholdConfig(max_pressure=45.0, static_duration=700.0),
            advanced=AdvancedConfig(enable_fatigue_awareness=True, fatigue_penalty_weight=0.3)
        )
        
        new_version = hot_env.update_reward_function(new_config.model_dump(), source="test")
        is_pass = new_version > 0 and hot_env.has_pending_update
        passed_count += int(is_pass)
        print_test_result("更新配置成功", is_pass, f"New version: {new_version}, Pending: {hot_env.has_pending_update}")
    except Exception as e:
        print_test_result("更新配置成功", False, f"Error: {e}")

    # 5.3 应用待定更新
    try:
        hot_env = HotReloadableEnv(MockEnv())
        
        old_version = hot_env.current_function.version
        hot_env.update_reward_function(get_default_config().model_dump(), source="test_apply")
        
        applied = hot_env.apply_pending_update()
        current_version = hot_env.current_function.version
        
        is_pass = applied and current_version != old_version and not hot_env.has_pending_update
        passed_count += int(is_pass)
        print_test_result("应用待定更新", is_pass, f"Applied: {applied}, New version: {current_version}")
    except Exception as e:
        print_test_result("应用待定更新", False, f"Error: {e}")

    # 5.4 历史记录
    try:
        hot_env = HotReloadableEnv(MockEnv())
        
        hot_env.update_reward_function(get_default_config().model_dump(), source="update_1")
        hot_env.update_reward_function(get_default_config().model_dump(), source="update_2")
        hot_env.apply_pending_update()
        
        history = hot_env.get_history(limit=10)
        is_pass = len(history) >= 2
        passed_count += int(is_pass)
        print_test_result("历史记录保存", is_pass, f"History entries: {len(history)}")
    except Exception as e:
        print_test_result("历史记录保存", False, f"Error: {e}")

    # 5.5 回滚功能
    try:
        hot_env = HotReloadableEnv(MockEnv())
        
        v1 = hot_env.current_function.version
        hot_env.update_reward_function(get_default_config().model_dump(), source="rollback_test")
        v2_target = hot_env._pending_function.version if hot_env._pending_function else 0
        
        rolled_back = hot_env.rollback(v1)
        is_pass = rolled_back and hot_env.has_pending_update
        passed_count += int(is_pass)
        print_test_result("回滚到指定版本", is_pass, f"Rollback to v{v1}: {rolled_back}")
    except Exception as e:
        print_test_result("回滚到指定版本", False, f"Error: {e}")

    # 5.6 无待定更新时 apply 返回 False
    try:
        hot_env = HotReloadableEnv(MockEnv())
        result = hot_env.apply_pending_update()
        is_pass = result is False
        passed_count += int(is_pass)
        print_test_result("无更新时 apply 返回 False", is_pass)
    except Exception as e:
        print_test_result("无更新时 apply 返回 False", False, f"Error: {e}")

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def test_api_integration():
    """测试 6: API 集成模拟"""
    print_separator("Test 6: API 集成模拟")

    import asyncio
    from api.routes.reward_config import (
        _current_config,
        _custom_presets,
        _config_version,
        list_all_presets,
        get_preset_detail,
        create_custom_preset,
        validate_config_endpoint,
        score_config,
        compare_configs_diff,
    )
    from api.reward_config import (
        RewardConfig,
        RewardWeights,
        ComfortConfig,
        ThresholdConfig,
        validate_config,
        get_default_config,
    )

    passed_count = 0
    total_tests = 6

    async def run_async_tests():
        nonlocal passed_count

        # 6.1 列出所有预设 (同步调用模拟)
        try:
            presets = await list_all_presets(category=None, page=1, size=20)
            is_pass = len(presets) >= 5
            passed_count += int(is_pass)
            print_test_result("列出所有预设", is_pass, f"Count: {len(presets)}")
        except Exception as e:
            print_test_result("列出所有预设", False, f"Error: {e}")

        # 6.2 获取内置预设详情
        try:
            detail = await get_preset_detail("balanced")
            is_pass = detail.is_builtin == True and detail.name == "balanced"
            passed_count += int(is_pass)
            print_test_result("获取内置预设详情", is_pass, f"Builtin: {detail.is_builtin}")
        except Exception as e:
            print_test_result("获取内置预设详情", False, f"Error: {e}")

        # 6.3 创建自定义预设
        try:
            custom = RewardConfig(
                config_name="my_custom_test",
                description="测试用自定义预设",
                weights=RewardWeights(comfort=1.2, pressure=0.9, static_penalty=0.6, energy=0.25),
                comfort=ComfortConfig(),
                thresholds=ThresholdConfig()
            )
            result = await create_custom_preset(custom)
            is_pass = result.is_builtin == False and result.name == "my_custom_test"
            passed_count += int(is_pass)
            print_test_result("创建自定义预设", is_pass, f"Name: {result.name}, Builtin: {result.is_builtin}")
        except Exception as e:
            print_test_result("创建自定义预设", False, f"Error: {e}")

        # 6.4 配置验证端点
        try:
            good_config = get_default_config()
            validation = await validate_config_endpoint(good_config)
            is_pass = validation.valid == True and validation.score >= 80
            passed_count += int(is_pass)
            print_test_result("配置验证端点", is_pass, f"Valid: {validation.valid}, Score: {validation.score:.1f}")
        except Exception as e:
            print_test_result("配置验证端点", False, f"Error: {e}")

        # 6.5 配置评分
        try:
            default_cfg = get_default_config()
            score_result = await score_config(default_cfg)
            is_pass = 0 <= score_result.score <= 100 and len(score_result.grade) > 0
            passed_count += int(is_pass)
            print_test_result("配置评分系统", is_pass, f"Score: {score_result.score:.1f}, Grade: {score_result.grade}")
        except Exception as e:
            print_test_result("配置评分系统", False, f"Error: {e}")

        # 6.6 配置差异对比
        try:
            base = get_default_config()
            modified = RewardConfig(
                config_name="modified",
                weights=RewardWeights(comfort=1.8, pressure=0.4, static_penalty=0.2, energy=0.15),
                comfort=ComfortConfig(),
                thresholds=ThresholdConfig(max_pressure=70.0, static_duration=1800.0)
            )
            diff = await compare_configs_diff(base, modified)
            is_pass = diff.summary['modified'] > 0
            passed_count += int(is_pass)
            print_test_result("配置差异对比", is_pass, f"Differences: {len(diff.differences)}, Summary: {diff.summary}")
        except Exception as e:
            print_test_result("配置差异对比", False, f"Error: {e}")

    asyncio.run(run_async_tests())

    print(f"\n  📊 结果: {passed_count}/{total_tests} 通过")
    return passed_count == total_tests


def main():
    """主测试函数"""
    print("\n" + "🎯" * 35)
    print("   人体工学座椅 AI 系统 - 奖励函数配置系统测试套件")
    print("   Reward Configuration System Test Suite v2.2.0")
    print("🎯" * 35)

    results = []

    results.append(("数据模型验证", test_data_models()))
    results.append(("预设管理", test_preset_management()))
    results.append(("奖励值计算", test_reward_calculation()))
    results.append(("动态构建器", test_dynamic_builder()))
    results.append(("热更新机制", test_hot_reload()))
    results.append(("API 集成模拟", test_api_integration()))

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
        print("  🎉 所有测试通过！奖励函数配置系统功能正常。")
        return 0
    else:
        print(f"  ⚠️  有 {total_tests - total_passed} 个测试组未通过，请检查上述失败项。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
