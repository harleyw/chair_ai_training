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

import os
import time
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from api.posture_classifier import PostureClassifier, PostureResult

logger = logging.getLogger(__name__)


class ChairAIService:
    """人体工学座椅 AI 服务核心类"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.model_loaded = False
        self.load_time = None
        self.start_time = time.time()
        self.inference_count = 0

        # ONNX 相关属性
        self.onnx_inf = None
        self.onnx_loaded = False

        # 姿态分类器（v2.0 新增）
        self.posture_classifier = PostureClassifier()

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        加载训练好的 PPO 模型
        
        Args:
            model_path: 模型文件路径 (.zip)
            
        Returns:
            是否加载成功
        """
        try:
            from stable_baselines3 import PPO
            
            logger.info(f"Loading model from {model_path}")
            self.model = PPO.load(model_path)
            self.model_path = model_path
            self.model_loaded = True
            self.load_time = datetime.now().isoformat()
            
            logger.info(f"Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            return False
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            包含模型详细信息的字典
        """
        if not self.model_loaded or not self.model:
            return {
                "model_name": "Not Loaded",
                "version": "N/A",
                "status": "not_loaded",
                "observation_space": 20,
                "action_space": 8,
                "algorithm": "PPO"
            }
        
        try:
            policy = self.model.policy
            
            info = {
                "model_name": os.path.basename(self.model_path) if self.model_path else "Unknown",
                "version": "1.0.1",
                "status": "loaded",
                "loaded_at": self.load_time,
                "training_info": {
                    "algorithm": "PPO (Proximal Policy Optimization)",
                    "policy_type": type(policy).__name__,
                    "learning_rate": getattr(self.model, 'learning_rate', 'N/A'),
                    "n_steps": getattr(self.model, 'n_steps', 'N/A'),
                    "batch_size": getattr(self.model, 'batch_size', 'N/A'),
                    "total_timesteps": getattr(self.model, '_num_timesteps', 0),
                },
                "observation_space": 20,
                "action_space": 8,
                "algorithm": "PPO"
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "model_name": "Error",
                "status": "error",
                "error": str(e)
            }
    
    def preprocess_sensor_data(self, sensor_data: Dict) -> np.ndarray:
        """
        预处理传感器数据为模型输入格式
        
        Args:
            sensor_data: 原始传感器数据字典
            
        Returns:
            20维观察向量
        """
        pressure_matrix = np.array(sensor_data['pressure_matrix'], dtype=np.float32)
        posture_angles = np.array(sensor_data['posture_angles'], dtype=np.float32)
        sitting_duration = float(sensor_data['sitting_duration'])
        user_weight = float(sensor_data['user_weight'])
        user_height = float(sensor_data['user_height'])
        fatigue_level = float(sensor_data.get('fatigue_level', 0.0))
        
        pressure_flat = pressure_matrix.flatten()
        
        pressure_mean = np.mean(pressure_flat)
        pressure_std = np.std(pressure_flat)
        pressure_max = np.max(pressure_flat)
        
        sitting_normalized = min(sitting_duration / 7200.0, 1.0)
        
        weight_normalized = (user_weight - 50.0) / 100.0
        height_normalized = (user_height - 1.4) / 1.0
        
        observation = np.zeros(20, dtype=np.float32)
        
        observation[0] = 0.5  
        observation[1] = 0.5  
        observation[2] = 0.5  
        observation[3] = 0.5  
        observation[4] = 0.5  
        observation[5] = 0.5  
        observation[6] = 0.5  
        observation[7] = 0.5  
        
        observation[8] = posture_angles[0] / 45.0
        observation[9] = posture_angles[1] / 30.0
        observation[10] = posture_angles[2] / 30.0
        
        observation[11] = pressure_mean
        observation[12] = pressure_std
        observation[13] = sitting_normalized
        
        observation[14] = weight_normalized
        observation[15] = height_normalized
        
        observation[16] = fatigue_level
        
        observation[17] = pressure_max
        observation[18] = np.percentile(pressure_flat, 90)
        observation[19] = np.sum(pressure_flat > 0.7) / len(pressure_flat)
        
        return observation
    
    def predict_action(self, sensor_data: Dict) -> Tuple[np.ndarray, float]:
        """
        预测座椅调整动作

        Args:
            sensor_data: 预处理后的传感器数据

        Returns:
            (动作向量, 置信度)
        """
        if not self.model_loaded and not self.onnx_loaded:
            logger.warning("Model not loaded, using rule-based fallback")
            return self._rule_based_action(sensor_data), 0.5

        try:
            # ONNX 推理分支（优先使用）
            if self.onnx_loaded and self.onnx_inf:
                try:
                    observation = self.preprocess_sensor_data(sensor_data)
                    action, confidence = self.onnx_inf.predict(observation)
                    self.inference_count += 1
                    return action, confidence
                except Exception as e:
                    logger.warning(f"ONNX inference error: {e}, falling back to PyTorch")

            # 原有的 PyTorch 推理代码
            if not self.model_loaded or not self.model:
                return self._rule_based_action(sensor_data), 0.5

            observation = self.preprocess_sensor_data(sensor_data)
            observation_batch = observation.reshape(1, -1)

            action, _states = self.model.predict(observation_batch, deterministic=True)

            action = np.clip(action, -1, 1).flatten()

            confidence = 0.85 + np.random.uniform(-0.1, 0.1)
            confidence = np.clip(confidence, 0, 1)

            self.inference_count += 1

            return action, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}, falling back to rules")
            return self._rule_based_action(sensor_data), 0.5
    
    def _rule_based_action(self, sensor_data: Dict) -> np.ndarray:
        """
        基于规则的回退策略（当模型未加载时使用）
        
        Args:
            sensor_data: 传感器数据
            
        Returns:
            规则基础的动作向量
        """
        action = np.zeros(8, dtype=np.float32)
        
        posture_angles = sensor_data.get('posture_angles', [0, 0, 0])
        sitting_duration = sensor_data.get('sitting_duration', 0)
        fatigue_level = sensor_data.get('fatigue_level', 0)
        pressure_matrix = np.array(sensor_data.get('pressure_matrix', [[0]*8]*8))
        
        head_angle = posture_angles[0] if len(posture_angles) > 0 else 0
        shoulder_angle = posture_angles[1] if len(posture_angles) > 1 else 0
        pelvis_angle = posture_angles[2] if len(posture_angles) > 2 else 0
        
        if abs(head_angle) > 20:
            action[4] = -np.sign(head_angle) * 0.3  
            action[5] = -np.sign(head_angle) * 0.2  
        
        if abs(shoulder_angle) > 15:
            action[1] = -np.sign(shoulder_angle) * 0.25 
        
        if abs(pelvis_angle) > 12:
            action[2] = -np.sign(pelvis_angle) * 0.3  
            action[3] = np.sign(pelvis_angle) * 0.2   
        
        if sitting_duration > 3600: 
            action[1] += 0.15  
            action[0] += 0.1   
        
        if fatigue_level > 0.7:
            action[3] += 0.2  
            action[1] += 0.1  
        
        pressure_imbalance = abs(np.mean(pressure_matrix[:, :4]) - np.mean(pressure_matrix[:, 4:]))
        if pressure_imbalance > 0.2:
            action[6] -= np.sign(np.mean(pressure_matrix[:, :4]) - np.mean(pressure_matrix[:, 4:])) * 0.2
            action[7] += np.sign(np.mean(pressure_matrix[:, :4]) - np.mean(pressure_matrix[:, 4:])) * 0.2
        
        action = np.clip(action, -1, 1)
        
        return action
    
    def analyze_posture(self, sensor_data: Dict) -> Dict:
        """
        分析姿态并生成建议

        Args:
            sensor_data: 传感器数据

        Returns:
            姿态分析结果字典 (analysis, issues, comfort_score, risk_level)
        """
        posture_angles = sensor_data.get('posture_angles', [0, 0, 0])
        sitting_duration = sensor_data.get('sitting_duration', 0)
        fatigue_level = sensor_data.get('fatigue_level', 0)
        pressure_matrix = np.array(sensor_data.get('pressure_matrix', [[0]*8]*8))

        head_angle = posture_angles[0]
        shoulder_angle = posture_angles[1]
        pelvis_angle = posture_angles[2]

        issues = []
        risk_level = "low"

        if abs(head_angle) > 25:
            issues.append("头部前倾/后仰过度，建议调整头枕位置")
            risk_level = "high"
        elif abs(head_angle) > 15:
            issues.append("头部角度偏差较大")
            risk_level = "medium"

        if abs(shoulder_angle) > 20:
            issues.append("肩部姿态不端正，建议调整靠背角度")
            risk_level = "high"
        elif abs(shoulder_angle) > 10:
            issues.append("肩部轻微倾斜")
            risk_level = max(risk_level, "medium")

        if abs(pelvis_angle) > 15:
            issues.append("骨盆倾斜严重，建议调整腰托支撑")
            risk_level = "high"
        elif abs(pelvis_angle) > 8:
            issues.append("骨盆有轻微偏移")
            risk_level = max(risk_level, "medium")

        if sitting_duration > 7200:
            issues.append(f"已连续静坐 {int(sitting_duration//3600)} 小时{int((sitting_duration%3600)//60)} 分钟，建议起身活动")
            risk_level = "high"
        elif sitting_duration > 3600:
            issues.append("静坐时间超过1小时，建议适当休息")
            risk_level = max(risk_level, "medium")

        if fatigue_level > 0.8:
            issues.append("疲劳度较高，建议增加腰部支撑")
            risk_level = "high"
        elif fatigue_level > 0.5:
            issues.append("出现轻度疲劳迹象")

        max_pressure = np.max(pressure_matrix)
        if max_pressure > 0.85:
            issues.append("局部压力过高，存在压疮风险")
            risk_level = "high"
        elif max_pressure > 0.7:
            issues.append("部分区域压力偏高")
            risk_level = max(risk_level, "medium")

        pressure_left = np.mean(pressure_matrix[:, :4])
        pressure_right = np.mean(pressure_matrix[:, 4:])
        imbalance = abs(pressure_left - pressure_right)
        if imbalance > 0.25:
            issues.append("左右压力分布不均，建议调整坐姿或扶手高度")
            risk_level = max(risk_level, "medium")

        comfort_score = self._calculate_comfort_score(
            head_angle, shoulder_angle, pelvis_angle,
            sitting_duration, fatigue_level, max_pressure, imbalance
        )

        analysis = {
            "head_posture": "normal" if abs(head_angle) <= 15 else ("slight" if abs(head_angle) <= 25 else "poor"),
            "shoulder_posture": "normal" if abs(shoulder_angle) <= 10 else ("slight" if abs(shoulder_angle) <= 20 else "poor"),
            "pelvis_posture": "normal" if abs(pelvis_angle) <= 8 else ("slight" if abs(pelvis_angle) <= 15 else "poor"),
            "sitting_duration_minutes": int(sitting_duration // 60),
            "fatigue_percentage": round(fatigue_level * 100, 1),
            "max_pressure_point": round(max_pressure, 3),
            "pressure_balance_score": round(1 - imbalance, 3),
            "overall_risk_level": risk_level,
            "issues_detected": len(issues),
            "comfort_score": round(comfort_score, 2)
        }

        # v2.0 新增：调用姿态分类器进行细粒度分类
        try:
            posture_result: PostureResult = self.posture_classifier.classify(sensor_data)
            
            # 将分类结果添加到 analysis 字典中（向后兼容）
            analysis["posture_detail"] = {
                "posture_type": posture_result.posture_type.value,
                "posture_name_cn": self.posture_classifier.get_posture_name_cn(posture_result.posture_type),
                "severity": posture_result.severity.value if isinstance(posture_result.severity, str) else posture_result.severity.value,
                "confidence": round(posture_result.confidence, 3),
                "risk_areas": posture_result.risk_areas,
                "recommended_exercises": posture_result.recommended_exercises,
                "primary_adjustments": posture_result.primary_adjustments,
                "message": posture_result.message
            }
            
            # 如果分类器检测到更严重的风险等级，更新风险级别
            severity_map = {"ideal": "low", "good": "low", "warning": "medium", "danger": "high"}
            classifier_risk = severity_map.get(
                posture_result.severity.value if isinstance(posture_result.severity, str) else posture_result.severity.value,
                "medium"
            )
            if classifier_risk == "high" or (classifier_risk == "medium" and risk_level == "low"):
                risk_level = classifier_risk
                analysis["overall_risk_level"] = risk_level
            
            # 添加分类器特有的问题到 issues 列表
            if posture_result.message and posture_result.posture_type.value != "normal":
                if posture_result.message not in issues:
                    issues.insert(0, posture_result.message)
                    
        except Exception as e:
            logger.warning(f"Posture classification error: {e}, using basic analysis only")
            analysis["posture_detail"] = None

        return analysis, issues, comfort_score, risk_level
    
    def _calculate_comfort_score(
        self,
        head_angle: float,
        shoulder_angle: float,
        pelvis_angle: float,
        duration: float,
        fatigue: float,
        max_pressure: float,
        imbalance: float
    ) -> float:
        """
        计算综合舒适度评分 (0-100)
        """
        score = 100.0
        
        score -= min(abs(head_angle) * 1.5, 30)
        score -= min(abs(shoulder_angle) * 1.2, 24)
        score -= min(abs(pelvis_angle) * 1.8, 27)
        
        score -= min(duration / 7200 * 20, 20)
        score -= fatigue * 15
        score -= max_pressure * 10
        score -= imbalance * 15
        
        return max(0, min(100, score))
    
    def load_onnx_model(self, model_path: str) -> bool:
        """
        加载 ONNX 模型用于推理

        Args:
            model_path: .onnx 文件路径

        Returns:
            是否加载成功
        """
        try:
            from export.runtime_inference import ONNXInference

            logger.info(f"Loading ONNX model from {model_path}")

            self.onnx_inf = ONNXInference(model_path)

            if self.onnx_inf.loaded:
                self.onnx_loaded = True
                logger.info("ONNX model loaded successfully")
                return True
            else:
                logger.warning("ONNX model loading failed")
                self.onnx_loaded = False
                return False

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.onnx_loaded = False
            return False

    def export_to_onnx(
        self,
        sb3_model_path: str,
        output_path: str,
        dynamic_batch: bool = False
    ) -> Dict:
        """
        导出当前模型到 ONNX 格式

        Args:
            sb3_model_path: Stable-Baselines3 模型路径
            output_path: 输出路径
            dynamic_batch: 是否动态 batch

        Returns:
            导出信息字典
        """
        try:
            from export.exporter import export_model

            result = export_model(
                model_path=sb3_model_path,
                output_path=output_path,
                dynamic_batch=dynamic_batch
            )

            return result

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_uptime(self) -> float:
        """获取服务运行时间（秒）"""
        return time.time() - self.start_time
    
    def get_stats(self) -> Dict:
        """获取服务统计信息"""
        return {
            "uptime_seconds": self.get_uptime(),
            "inference_count": self.inference_count,
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "onnx_loaded": getattr(self, 'onnx_loaded', False),
            "onnx_model_path": getattr(self, 'onnx_inf', None).model_path if getattr(self, 'onnx_inf', None) else None
        }
    
    # ========== WebSocket 流式处理方法 ==========
    
    def process_stream(
        self,
        sensor_data: Dict,
        session_history: Optional[List[Dict]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        处理实时传感器数据流（WebSocket 专用）
        
        与 predict_action 类似，但增加了历史上下文分析能力，
        可用于检测姿态突变、趋势变化等
        
        Args:
            sensor_data: 当前传感器数据
            session_history: 该会话的历史数据列表（可选）
            
        Returns:
            (action_vector, confidence) 动作向量和置信度
        """
        # 基础推理
        action, confidence = self.predict_action(sensor_data)
        
        # 如果有历史数据，进行增强分析
        if session_history and len(session_history) > 5:
            action = self._analyze_trend(sensor_data, session_history, action)
            
            # 检测姿态突变时增加置信度调整
            if self._detect_posture_change(sensor_data, session_history):
                confidence = min(confidence * 1.1, 0.99)
                logger.debug("Posture change detected, adjusting confidence")
        
        return action, confidence
    
    def _analyze_trend(
        self,
        current_data: Dict,
        history: List[Dict],
        base_action: np.ndarray
    ) -> np.ndarray:
        """
        分析历史趋势并调整动作
        
        基于最近的数据趋势，平滑动作输出，避免频繁抖动
        
        Args:
            current_data: 当前传感器数据
            history: 历史消息列表
            base_action: 基础推理结果
            
        Returns:
            调整后的动作向量
        """
        if len(history) < 3:
            return base_action
        
        # 提取最近的姿态角度
        recent_angles = []
        for h in history[-5:]:
            data = h.get("data", {})
            angles = data.get("posture_angles", [0, 0, 0])
            recent_angles.append(angles)
        
        if len(recent_angles) < 3:
            return base_action
        
        # 计算姿态变化率
        angles_array = np.array(recent_angles)
        angle_changes = np.diff(angles_array, axis=0)
        avg_change = np.mean(np.abs(angle_changes), axis=0)
        
        # 如果姿态稳定，轻微增强当前动作；如果快速变化，减弱动作幅度
        stability_factor = 1.0 - np.clip(avg_change / 10.0, 0, 0.3)
        
        adjusted_action = base_action * (0.8 + 0.2 * stability_factor.mean())
        
        return adjusted_action
    
    def _detect_posture_change(
        self,
        current_data: Dict,
        history: List[Dict]
    ) -> bool:
        """
        检测是否发生姿态突变
        
        Args:
            current_data: 当前传感器数据
            history: 历史数据
            
        Returns:
            是否检测到突变
        """
        if len(history) < 3:
            return False
        
        current_angles = current_data.get("posture_angles", [0, 0, 0])
        
        # 计算与历史的平均偏差
        deviations = []
        for h in history[-3:]:
            data = h.get("data", {})
            hist_angles = data.get("posture_angles", current_angles)
            
            deviation = sum(abs(c - h) for c, h in zip(current_angles, hist_angles))
            deviations.append(deviation)
        
        avg_deviation = np.mean(deviations) if deviations else 0
        
        # 阈值：平均偏差 > 15 度视为突变
        return avg_deviation > 15
    
    def detect_anomalies(
        self,
        sensor_data: Dict,
        session_history: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        实时异常检测
        
        检测多种异常情况：
        - 姿态突变
        - 疲劳累积超阈值
        - 压力分布不均
        - 静坐时间过长
        
        Args:
            sensor_data: 当前传感器数据
            session_history: 会话历史
            
        Returns:
            异常列表，每个元素包含 type, severity, message
        """
        anomalies = []
        
        posture_angles = sensor_data.get("posture_angles", [0, 0, 0])
        sitting_duration = sensor_data.get("sitting_duration", 0)
        fatigue_level = sensor_data.get("fatigue_level", 0)
        pressure_matrix = np.array(sensor_data.get("pressure_matrix", [[0]*8]*8))
        
        # 1. 姿态突变检测
        if session_history and len(session_history) >= 2:
            if self._detect_posture_change(sensor_data, session_history):
                anomalies.append({
                    "type": "posture_sudden_change",
                    "severity": "warning",
                    "message": "检测到姿态突变",
                    "value": max(abs(a) for a in posture_angles)
                })
        
        # 2. 疲劳度告警
        if fatigue_level > 0.7:
            anomalies.append({
                "type": "fatigue_high",
                "severity": "warning" if fatigue_level < 0.9 else "critical",
                "message": f"疲劳度过高 ({fatigue_level*100:.0f}%)",
                "value": fatigue_level
            })
        
        # 3. 压力峰值检测
        max_pressure = np.max(pressure_matrix)
        if max_pressure > 0.85:
            anomalies.append({
                "type": "pressure_peak",
                "severity": "warning" if max_pressure < 0.95 else "critical",
                "message": f"局部压力过高 ({max_pressure*100:.0f}%)",
                "value": float(max_pressure)
            })
        
        # 4. 压力不平衡检测
        if pressure_matrix.shape == (8, 8):
            left_pressure = np.mean(pressure_matrix[:, :4])
            right_pressure = np.mean(pressure_matrix[:, 4:])
            imbalance = abs(left_pressure - right_pressure)
            
            if imbalance > 0.25:
                anomalies.append({
                    "type": "pressure_imbalance",
                    "severity": "info",
                    "message": "左右压力分布不均",
                    "value": float(imbalance)
                })
        
        # 5. 静坐时间过长
        if sitting_duration > 7200:  # 2小时
            anomalies.append({
                "type": "sitting_too_long",
                "severity": "critical" if sitting_duration > 10800 else "warning",
                "message": f"静坐时间过长 ({int(sitting_duration//3600)}h{int((sitting_duration%3600)//60)}m)",
                "value": sitting_duration
            })
        
        return anomalies
    
    def get_stream_status(self) -> Dict:
        """获取流式处理状态"""
        return {
            "streaming_enabled": True,
            "anomaly_detection_enabled": True,
            "trend_analysis_enabled": True,
            "supported_features": [
                "realtime_inference",
                "posture_change_detection",
                "fatigue_monitoring",
                "pressure_analysis",
                "trend_smoothing"
            ]
        }
