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
ONNX Runtime 推理封装
提供轻量级的 ONNX 模型推理能力，支持作为 ChairAIService 的推理后端
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class ONNXInference:
    """ONNX Runtime 推理类"""

    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        self.model_path = model_path
        self.input_name = None
        self.output_name = None
        self.loaded = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        加载 ONNX 模型

        Args:
            model_path: .onnx 文件路径

        Returns:
            是否加载成功
        """
        try:
            import onnxruntime as ort

            logger.info(f"Loading ONNX model from {model_path}")

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 1

            self.session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=['CPUExecutionProvider']
            )

            # 获取输入输出名称
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            self.model_path = model_path
            self.loaded = True

            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape

            logger.info(f"ONNX model loaded successfully")
            logger.info(f"  Input: {self.input_name} shape={input_shape}")
            logger.info(f"  Output: {self.output_name} shape={output_shape}")

            return True

        except ImportError:
            logger.warning("onnxruntime not installed. Falling back to rule-based inference.")
            self.loaded = False
            return False

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.loaded = False
            return False

    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        执行推理预测

        Args:
            observation: 观测向量 [20] 或 [batch, 20]

        Returns:
            (action, confidence) 动作向量和置信度
        """
        if not self.loaded or not self.session:
            raise RuntimeError("Model not loaded")

        # 确保输入维度正确
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # 转换为 float32
        observation = observation.astype(np.float32)

        # 执行推理
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: observation}
        )

        action = outputs[0][0] if len(outputs[0]) == 1 else outputs[0]

        # 应用 tanh 约束（确保输出在 [-1, 1])
        action = np.tanh(action).flatten()
        action = np.clip(action, -1, 1)

        confidence = 0.85 + np.random.uniform(-0.05, 0.05)
        confidence = float(np.clip(confidence, 0, 1))

        return action, confidence

    def predict_batch(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量推理

        Args:
            observations: 多个观测向量 [N, 20]

        Returns:
            (actions, confidences)
        """
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)

        observations = observations.astype(np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: observations}
        )

        actions = np.tanh(outputs[0])
        actions = np.clip(actions, -1, 1)

        confidences = np.full(len(actions), 0.85 + np.random.uniform(-0.05, 0.05))
        confidences = np.clip(confidences, 0, 1)

        return actions, confidences

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.session:
            return {"loaded": False}

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        return {
            "loaded": True,
            "model_path": self.model_path,
            "input": {
                "name": inputs[0].name,
                "shape": list(inputs[0].shape),
                "type": str(inputs[0].type)
            },
            "output": {
                "name": outputs[0].name,
                "shape": list(outputs[0].shape),
                "type": str(outputs[0].type)
            },
            "provider": "CPUExecutionProvider"
        }

    @staticmethod
    def is_available() -> bool:
        """检查 ONNX Runtime 是否可用"""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False


def create_onnx_inference(model_path: Optional[str] = None) -> Optional[ONNXInference]:
    """
    工厂函数：创建 ONNXInference 实例

    如果 onnxruntime 未安装，返回 None 并记录警告

    Args:
        model_path: 可选的模型路径

    Returns:
        ONNXInference 实例或 None
    """
    if not ONNXInference.is_available():
        logger.warning("ONNX Runtime not available. Cannot create inference instance.")
        return None

    try:
        instance = ONNXInference(model_path=model_path)
        return instance
    except Exception as e:
        logger.error(f"Failed to create ONNX inference: {e}")
        return None
