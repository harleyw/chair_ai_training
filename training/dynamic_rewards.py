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
动态奖励函数构建器和加载器
支持从配置对象构建可调用的奖励函数，实现运行时参数调整
"""

import math
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from api.reward_config import (
    RewardConfig,
    RewardBreakdown,
    get_default_config,
    get_preset,
    validate_config,
)


class RewardFunctionResult:
    """奖励函数计算结果"""

    def __init__(
        self,
        total_reward: float,
        breakdown: RewardBreakdown,
        config_version: int = 0
    ):
        self.total_reward = total_reward
        self.breakdown = breakdown
        self.config_version = config_version

    def __repr__(self):
        return f"RewardFunctionResult(total={self.total_reward:.4f}, version={self.config_version})"


class DynamicRewardFunction:
    """
    动态奖励函数
    
    从 RewardConfig 配置构建的可调用对象，
    支持热更新和版本追踪。
    
    Usage:
        builder = RewardFunctionBuilder()
        reward_fn = builder.build_from_config(config_dict)
        
        result = reward_fn(
            action_magnitude=0.5,
            spine_curvature=0.2,
            pressure_variance=0.15,
            max_pressure=40.0,
            static_duration=600.0,
            fatigue_level=0.4
        )
        
        print(result.total_reward)
        print(result.breakdown.comfort_component)
    """

    def __init__(
        self,
        config: RewardConfig,
        version: int = 1
    ):
        self._config = config
        self._version = version
        self._lock = threading.RLock()
        self._call_count = 0
        self._created_at = datetime.now()

    @property
    def config(self) -> RewardConfig:
        return self._config

    @property
    def version(self) -> int:
        return self._version

    @property
    def call_count(self) -> int:
        return self._call_count

    def __call__(
        self,
        action_magnitude: float,
        spine_curvature: float = 0.0,
        pressure_variance: float = 0.0,
        max_pressure: float = 30.0,
        static_duration: float = 0.0,
        fatigue_level: float = 0.0,
        posture_changed: bool = False,
        symmetry_ratio: float = 1.0,
        **kwargs
    ) -> RewardFunctionResult:
        """
        计算奖励值
        
        Args:
            action_magnitude: 动作幅度 (必须)
            spine_curvature: 脊柱曲率偏差 (默认 0.0)
            pressure_variance: 压力分布方差 (默认 0.0)
            max_pressure: 最大压力点 (默认 30.0)
            static_duration: 静态时长秒数 (默认 0.0)
           疲劳程度 0-1 (默认 0.0)
            posture_changed: 姿态是否变化 (默认 False)
            symmetry_ratio: 左右对称比例 0-1 (默认 1.0)
            **kwargs: 其他自定义参数
            
        Returns:
            RewardFunctionResult 包含总奖励和分解明细
        """
        with self._lock:
            self._call_count += 1

        from api.reward_config import calculate_reward_breakdown

        breakdown = calculate_reward_breakdown(
            config=self._config,
            action_magnitude=action_magnitude,
            spine_curvature=spine_curvature,
            pressure_variance=pressure_variance,
            max_pressure=max_pressure,
            static_duration=static_duration,
            fatigue_level=fatigue_level,
            posture_changed=posture_changed,
            symmetry_ratio=symmetry_ratio
        )

        if self._config.advanced and self._config.advanced.custom_formula:
            try:
                custom_result = self._execute_custom_formula(
                    action_magnitude=action_magnitude,
                    spine_curvature=spine_curvature,
                    pressure_variance=pressure_variance,
                    max_pressure=max_pressure,
                    static_duration=static_duration,
                    fatigue_level=fatigue_level,
                    posture_changed=posture_changed,
                    symmetry_ratio=symmetry_ratio,
                    **kwargs
                )
                
                if isinstance(custom_result, (int, float)):
                    breakdown.total_reward = float(custom_result)
            except Exception as e:
                pass

        return RewardFunctionResult(
            total_reward=breakdown.total_reward,
            breakdown=breakdown,
            config_version=self._version
        )

    def _execute_custom_formula(
        self,
        **variables
    ) -> Optional[float]:
        """
        安全执行自定义公式
        
        在受限环境中执行用户提供的 Python 表达式。
        只允许使用数学函数，禁止任何危险操作。
        
        Args:
            **variables: 公式可用的变量
            
        Returns:
            公式计算结果，或 None (如果执行失败)
        """
        formula = self._config.advanced.custom_formula
        if not formula:
            return None

        safe_globals = {
            "__builtins__": {},
            "math": math,
            "np": np,
            "abs": abs,
            "max": max,
            "min": min,
            "round": round,
            "pow": pow,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log": math.log,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }

        safe_locals = {
            k: v for k, v in variables.items()
        }

        try:
            result = eval(formula, safe_globals, safe_locals)

            if isinstance(result, (int, float, np.number)):
                return float(result)
            else:
                return None
                
        except Exception as e:
            return None


class RewardFunctionBuilder:
    """
    奖励函数构建器工厂
    
    用于从配置字典或预设名称构建 DynamicRewardFunction 实例。
    
    Usage:
        builder = RewardFunctionBuilder()
        
        # 从配置字典构建
        fn1 = builder.build_from_config(config_dict)
        
        # 从预设名称构建
        fn2 = builder.build_from_preset("balanced")
        
        # 构建默认配置的函数
        fn3 = builder.build_default()
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._version_counter = 0
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._version_counter = 0
            self._initialized = True

    def build_from_config(
        self,
        config_dict: Dict[str, Any],
        validate: bool = True
    ) -> DynamicRewardFunction:
        """
        从配置字典构建奖励函数
        
        Args:
            config_dict: 配置字典
            validate: 是否验证配置 (默认 True)
            
        Returns:
            DynamicRewardFunction 实例
            
        Raises:
            ValueError: 如果 validate=True 且配置无效
        """
        if validate:
            validation = validate_config(config_dict)
            if not validation.valid:
                error_msgs = [e.get('message', str(e)) for e in validation.errors]
                raise ValueError(f"配置验证失败: {'; '.join(error_msgs)}")

        config = RewardConfig(**config_dict)

        with self._lock:
            self._version_counter += 1
            version = self._version_counter

        return DynamicRewardFunction(config=config, version=version)

    def build_from_preset(
        self,
        preset_name: str
    ) -> DynamicRewardFunction:
        """
        从预设名称构建奖励函数
        
        Args:
            preset_name: 预设名称 (balanced/health_first/comfort_priority/strict_posture/energy_saving)
            
        Returns:
            DynamicRewardFunction 实例
            
        Raises:
            ValueError: 如果预设不存在
        """
        preset_config = get_preset(preset_name)

        if preset_config is None:
            available = list(get_preset(p) for p in ['balanced', 'health_first', 'comfort_priority', 'strict_posture', 'energy_saving'] if get_preset(p) is not None)
            raise ValueError(
                f"预设 '{preset_name}' 不存在。可用预设: {[p for p in ['balanced', 'health_first', 'comfort_priority', 'strict_posture', 'energy_saving'] if get_preset(p)]}"
            )

        with self._lock:
            self._version_counter += 1
            version = self._version_counter

        return DynamicRewardFunction(config=preset_config, version=version)

    def build_default(self) -> DynamicRewardFunction:
        """
        构建默认配置的奖励函数
        
        Returns:
            使用系统默认配置的 DynamicRewardFunction 实例
        """
        default_config = get_default_config()

        with self._lock:
            self._version_counter += 1
            version = self._version_counter

        return DynamicRewardFunction(config=default_config, version=version)


class ConfigHistoryEntry:
    """配置历史条目"""

    def __init__(
        self,
        version: int,
        config: RewardConfig,
        applied_at: datetime,
        source: str = "manual"
    ):
        self.version = version
        self.config = config
        self.applied_at = applied_at
        self.source = source


class HotReloadableEnv:
    """
    支持热更新的环境包装器
    
    包装 ChairEnv 或类似的环境实例，
    允许在训练过程中动态更新奖励函数而不中断当前 episode。
    
    Usage:
        env = ChairEnv(...)
        hot_env = HotReloadableEnv(env)
        
        # 开始训练...
        # 在某个时间点更新配置
        hot_env.update_reward_function(new_config)
        
        # 当前 episode 继续使用旧函数
        # 下次 env.reset() 时自动应用新函数
    """

    MAX_HISTORY_SIZE = 10

    def __init__(self, env, initial_config: Optional[RewardConfig] = None):
        self._env = env
        self._builder = RewardFunctionBuilder()

        if initial_config:
            self._current_function = self._builder.build_from_config(initial_config.model_dump())
        else:
            self._current_function = self._builder.build_default()

        self._pending_function: Optional[DynamicRewardFunction] = None
        self._history: List[ConfigHistoryEntry] = []
        self._dirty_flag = False
        self._lock = threading.RLock()

        self._record_history(self._current_function.config, source="initial")

    @property
    def current_function(self) -> DynamicRewardFunction:
        return self._current_function

    @property
    def current_config(self) -> RewardConfig:
        return self._current_function.config

    @property
    def has_pending_update(self) -> bool:
        return self._pending_function is not None

    @property
    def history_size(self) -> int:
        return len(self._history)

    def update_reward_function(
        self,
        new_config: Dict[str, Any] | RewardConfig,
        source: str = "api"
    ) -> int:
        """
        更新奖励函数（下次 reset 后生效）
        
        Args:
            new_config: 新配置 (dict 或 RewardConfig 对象)
            source: 更新来源标识
            
        Returns:
            新配置的版本号
            
        Note:
            正在进行中的 episode 会继续使用当前函数，
            下次调用 apply_pending_update() 或 reset() 时应用新函数。
        """
        with self._lock:
            if isinstance(new_config, dict):
                new_fn = self._builder.build_from_config(new_config)
            elif isinstance(new_config, RewardConfig):
                new_fn = self._builder.build_from_config(new_config.model_dump())
            else:
                raise TypeError("new_config 必须是 dict 或 RewardConfig")

            self._pending_function = new_fn
            self._dirty_flag = True

            self._record_history(new_fn.config, source=source)

            logger = __import__('logging').getLogger(__name__)
            logger.info(
                f"Pending reward function update scheduled: "
                f"version={new_fn.version}, source={source}"
            )

            return new_fn.version

    def apply_pending_update(self) -> bool:
        """
        应用待定的更新
        
        将 pending 的奖励函数切换为当前生效的函数。
        通常在 episode 结束后或 env.reset() 时调用。
        
        Returns:
            是否成功应用了更新
        """
        with self._lock:
            if self._pending_function is None:
                return False

            old_version = self._current_function.version
            self._current_function = self._pending_function
            self._pending_function = None
            self._dirty_flag = False

            new_version = self._current_function.version

            logger = __import__('logging').getLogger(__name__)
            logger.info(
                f"Reward function updated: {old_version} -> {new_version}"
            )

            return True

    def calculate_reward(self, **kwargs) -> RewardFunctionResult:
        """
        使用当前奖励函数计算奖励值
        
        Args:
            **kwargs: 奖励函数所需的参数
            
        Returns:
            RewardFunctionResult
        """
        return self._current_function(**kwargs)

    def rollback(self, target_version: int) -> bool:
        """
        回滚到指定版本的配置
        
        Args:
            target_version: 目标版本号
            
        Returns:
            是否回滚成功
        """
        with self._lock:
            for entry in reversed(self._history):
                if entry.version == target_version:
                    self._pending_function = self._builder.build_from_config(
                        entry.config.model_dump()
                    )
                    self._dirty_flag = True
                    return True

            return False

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取配置变更历史
        
        Args:
            limit: 返回的最大条目数
            
        Returns:
            历史记录列表
        """
        with self._lock:
            recent = self._history[-limit:]
            return [
                {
                    "version": entry.version,
                    "config_name": entry.config.config_name,
                    "applied_at": entry.applied_at.isoformat(),
                    "source": entry.source
                }
                for entry in recent
            ]

    def _record_history(self, config: RewardConfig, source: str = "unknown"):
        """记录配置到历史"""
        with self._lock:
            version = self._current_function.version if hasattr(self, '_current_function') else 0

            entry = ConfigHistoryEntry(
                version=version + 1 if self._pending_function else version,
                config=config.model_copy(deep=True),
                applied_at=datetime.now(),
                source=source
            )

            self._history.append(entry)

            if len(self._history) > self.MAX_HISTORY_SIZE:
                self._history = self._history[-self.MAX_HISTORY_SIZE:]
