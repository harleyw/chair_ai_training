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
ONNX 模型导出工具
将 Stable-Baselines3 PPO 模型转换为 ONNX 格式
"""

import os
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ONNXExporter:
    """ONNX 导出器类"""
    
    def __init__(self, opset_version: int = 17):
        self.opset_version = opset_version
    
    def load_sb3_model(self, model_path: str):
        """
        加载 Stable-Baselines3 PPO 模型
        
        Args:
            model_path: .zip 文件路径
            
        Returns:
            (model, policy_network) 元组
        """
        from stable_baselines3 import PPO
        
        logger.info(f"Loading SB3 model from {model_path}")
        model = PPO.load(model_path)
        
        logger.info("Extracting actor policy network")
        policy = model.policy
        
        return model, policy
    
    def export_to_onnx(
        self,
        model_path: str,
        output_path: str,
        dynamic_batch: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        将 PPO 模型导出为 ONNX 格式
        
        Args:
            model_path: 输入的 .zip 模型路径
            output_path: 输出的 .onnx 文件路径
            dynamic_batch: 是否使用动态 batch 维度
            verbose: 是否输出详细信息
            
        Returns:
            包含导出信息的字典
        """
        import torch
        
        # 1. 加载模型
        model, policy = self.load_sb3_model(model_path)
        
        # 2. 设置为评估模式
        policy.eval()
        
        # 3. 创建示例输入 [batch_size=1, observation_dim=20]
        dummy_input = torch.randn(1, 20)
        
        # 4. 配置动态轴
        if dynamic_batch:
            dynamic_axes = {
                'observation': {0: 'batch_size'},
                'action': {0: 'batch_size'}
            }
        else:
            dynamic_axes = None
        
        # 5. 导出
        logger.info(f"Exporting to ONNX format: {output_path}")
        
        torch.onnx.export(
            policy,
            dummy_input,
            output_path,
            input_names=['observation'],
            output_names=['action'],
            dynamic_axes=dynamic_axes,
            opset_version=self.opset_version,
            do_constant_folding=True,
            verbose=verbose
        )
        
        # 6. 收集元数据
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        export_info = {
            "success": True,
            "output_path": output_path,
            "file_size_mb": round(file_size, 2),
            "opset_version": self.opset_version,
            "dynamic_batch": dynamic_batch,
            "input_shape": ["batch_size", "20"] if dynamic_batch else ["1", "20"],
            "output_shape": ["batch_size", "8"] if dynamic_batch else ["1", "8"],
            "model_architecture": "MLPPolicy(20→256→128→64→8)",
            "exported_at": datetime.now().isoformat(),
            "source_model": model_path
        }
        
        logger.info(f"Export completed successfully: {output_path} ({file_size:.2f} MB)")
        
        return export_info


def export_model(
    model_path: str,
    output_path: Optional[str] = None,
    dynamic_batch: bool = False,
    opset_version: int = 17
) -> Dict[str, Any]:
    """
    便捷函数：导出模型到 ONNX 格式
    
    Args:
        model_path: 输入模型路径
        output_path: 输出路径（默认自动生成）
        dynamic_batch: 是否动态 batch
        opset_version: ONNX OpSet 版本
        
    Returns:
        导出信息字典
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{base_name}.onnx"
    
    exporter = ONNXExporter(opset_version=opset_version)
    return exporter.export_to_onnx(
        model_path=model_path,
        output_path=output_path,
        dynamic_batch=dynamic_batch
    )
