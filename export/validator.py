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
ONNX 模型验证工具
验证导出的 ONNX 模型的结构、数值一致性和推理性能
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ONNXValidator:
    """ONNX 模型验证器"""

    def __init__(self):
        self.results = {}

    def validate_structure(self, onnx_path: str) -> Dict[str, Any]:
        """
        验证模型结构

        Args:
            onnx_path: ONNX 文件路径

        Returns:
            结构验证结果
        """
        try:
            import onnx

            logger.info(f"Validating ONNX model structure: {onnx_path}")

            model = onnx.load(onnx_path)

            # 验证模型
            onnx.checker.check_model(model)

            # 获取输入输出信息
            inputs = [inp for inp in model.graph.input]
            outputs = [out for out in model.graph.output]

            input_info = []
            for inp in inputs:
                shape = [dim.dim_value if dim.dim_value else dim.dim_param for dim in inp.type.tensor_type.shape.dim]
                input_info.append({
                    "name": inp.name,
                    "shape": shape,
                    "type": str(inp.type.tensor_type.elem_type)
                })

            output_info = []
            for out in outputs:
                shape = [dim.dim_value if dim.dim_value else dim.dim_param for dim in out.type.tensor_type.shape.dim]
                output_info.append({
                    "name": out.name,
                    "shape": shape,
                    "type": str(out.type.tensor_type.elem_type)
                })

            # 统计算子数量
            op_count = len(model.graph.node)
            op_types = set([node.op_type for node in model.graph.node])

            result = {
                "valid": True,
                "inputs": input_info,
                "outputs": output_info,
                "op_count": op_count,
                "op_types": list(op_types),
                "errors": [],
                "warnings": []
            }

            # 检查维度是否符合预期
            if input_info:
                expected_input_dim = 20
                actual_dim = input_info[0]["shape"][-1] if isinstance(input_info[0]["shape"][-1], int) else None

                if actual_dim and actual_dim != expected_input_dim:
                    result["warnings"].append(
                        f"Input dimension {actual_dim} differs from expected {expected_input_dim}"
                    )

            if output_info:
                expected_output_dim = 8
                actual_dim = output_info[0]["shape"][-1] if isinstance(output_info[0]["shape"][-1], int) else None

                if actual_dim and actual_dim != expected_output_dim:
                    result["warnings"].append(
                        f"Output dimension {actual_dim} differs from expected {expected_output_dim}"
                    )

            self.results["structure"] = result
            logger.info(f"Structure validation passed: {op_count} operators found")

            return result

        except Exception as e:
            result = {
                "valid": False,
                "error": str(e),
                "errors": [str(e)],
                "warnings": []
            }
            self.results["structure"] = result
            logger.error(f"Structure validation failed: {e}")
            return result

    def validate_numerical_consistency(
        self,
        sb3_model_path: str,
        onnx_path: str,
        num_samples: int = 100,
        tolerance: float = 1e-5
    ) -> Dict[str, Any]:
        """
        验证数值一致性：对比 PyTorch 和 ONNX Runtime 输出

        Args:
            sb3_model_path: Stable-Baselines3 模型路径
            onnx_path: ONNX 模型路径
            num_samples: 测试样本数
            tolerance: 允许的误差阈值

        Returns:
            数值一致性验证结果
        """
        import torch
        from stable_baselines3 import PPO
        from export.runtime_inference import ONNXInference

        logger.info("Validating numerical consistency...")

        # 加载 PyTorch 模型
        pytorch_model = PPO.load(sb3_model_path)
        policy = pytorch_model.policy
        policy.eval()

        # 加载 ONNX 模型
        onnx_inf = ONNXInference(onnx_path)

        if not onnx_inf.loaded:
            return {
                "passed": False,
                "error": "Failed to load ONNX model",
                "max_error": float('inf'),
                "mean_error": float('inf'),
                "samples_tested": 0
            }

        errors = []

        with torch.no_grad():
            for i in range(num_samples):
                # 生成随机输入
                test_input = np.random.randn(1, 20).astype(np.float32)

                # PyTorch 推理
                torch_input = torch.from_numpy(test_input)
                torch_output, _ = policy(torch_input)
                torch_action = torch_output.cpu().numpy().flatten()

                # ONNX 推理
                onnx_action, _ = onnx_inf.predict(test_input)

                # 计算误差
                error = np.abs(torch_action - onnx_action)
                max_error = np.max(error)
                mean_error = np.mean(error)

                errors.append(max_error)

                if max_error > tolerance:
                    logger.warning(
                        f"Sample {i}: max_error={max_error:.2e} exceeds tolerance={tolerance:.2e}"
                    )

        max_error = max(errors)
        mean_error = np.mean(errors)
        p95_error = np.percentile(errors, 95)
        p99_error = np.percentile(errors, 99)

        passed = max_error < tolerance

        result = {
            "passed": passed,
            "tolerance": tolerance,
            "max_error": float(max_error),
            "mean_error": float(mean_error),
            "p95_error": float(p95_error),
            "p99_error": float(p99_error),
            "samples_tested": num_samples,
            "all_within_tolerance": all(e < tolerance for e in errors)
        }

        self.results["numerical"] = result

        status = "PASSED" if passed else "FAILED"
        logger.info(f"Numerical validation {status}: max_error={max_error:.2e}, mean={mean_error:.2e}")

        return result

    def benchmark_performance(
        self,
        onnx_path: str,
        num_iterations: int = 1000,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        性能基准测试

        Args:
            onnx_path: ONNX 模型路径
            num_iterations: 迭代次数
            batch_size: 批量大小

        Returns:
            性能统计结果
        """
        from export.runtime_inference import ONNXInference

        logger.info(f"Running performance benchmark ({num_iterations} iterations)...")

        inf = ONNXInference(onnx_path)

        if not inf.loaded:
            return {
                "error": "Failed to load ONNX model",
                "throughput": 0,
                "latency_ms": {}
            }

        latencies = []

        # Warmup
        warmup_input = np.random.randn(batch_size, 20).astype(np.float32)
        for _ in range(10):
            inf.predict(warmup_input)

        # Benchmark
        for i in range(num_iterations):
            test_input = np.random.randn(batch_size, 20).astype(np.float32)

            start = time.perf_counter()
            inf.predict(test_input)
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        latencies = np.array(latencies)

        total_time = sum(latencies)
        throughput = num_iterations / (total_time / 1000)  # samples per second

        result = {
            "iterations": num_iterations,
            "batch_size": batch_size,
            "throughput_fps": round(throughput, 2),
            "latency_ms": {
                "mean": round(float(np.mean(latencies)), 4),
                "std": round(float(np.std(latencies)), 4),
                "min": round(float(np.min(latencies)), 4),
                "max": round(float(np.max(latencies)), 4),
                "p50": round(float(np.percentile(latencies, 50)), 4),
                "p95": round(float(np.percentile(latencies, 95)), 4),
                "p99": round(float(np.percentile(latencies, 99)), 4)
            },
            "total_time_ms": round(total_time, 2)
        }

        self.results["performance"] = result

        logger.info(
            f"Benchmark complete: throughput={throughput:.0f} fps, "
            f"mean_latency={np.mean(latencies):.4f}ms"
        )

        return result

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成完整验证报告

        Args:
            output_path: 可选的报告输出路径（JSON）

        Returns:
            完整报告字典
        """
        report = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "validation_results": self.results,
            "overall_status": "PASS" if all(
                r.get("valid", r.get("passed", True))
                for r in self.results.values()
            ) else "FAIL"
        }

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Report saved to {output_path}")

        return report


def validate_onnx_model(
    onnx_path: str,
    sb3_model_path: Optional[str] = None,
    run_benchmark: bool = True,
    output_report: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：完整验证流程

    Args:
        onnx_path: ONNX 模型路径
        sb3_model_path: SB3 模型路径（用于数值验证，可选）
        run_benchmark: 是否运行性能基准测试
        output_report: 报告输出路径

    Returns:
        完整验证结果
    """
    validator = ONNXValidator()

    # 1. 结构验证
    structure_result = validator.validate_structure(onnx_path)

    if not structure_result.get("valid"):
        report = validator.generate_report(output_report)
        return report

    # 2. 数值一致性验证（如果提供了 SB3 模型）
    if sb3_model_path and os.path.exists(sb3_model_path):
        validator.validate_numerical_consistency(sb3_model_path, onnx_path)

    # 3. 性能基准测试
    if run_benchmark:
        validator.benchmark_performance(onnx_path)

    # 4. 生成报告
    return validator.generate_report(output_report)
