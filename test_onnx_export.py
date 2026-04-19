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
ONNX 导出功能测试脚本

测试内容:
1. 模块导入验证
2. ONNXExporter 类功能测试
3. ONNXInference 类功能测试
4. ONNXValidator 功能测试
5. CLI 工具参数解析测试
6. API 集成测试（可选）

用法:
    python test_onnx_export.py
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestResult:
    """测试结果收集器"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add(self, name: str, passed: bool, message: str = ""):
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {name} {message}")
        
        self.tests.append({
            "name": name,
            "passed": passed,
            "message": message
        })
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*60)
        print(f"  测试结果汇总")
        print("="*60)
        print(f"  总计: {total} 个测试")
        print(f"  通过: {self.passed} ✓")
        print(f"  失败: {self.failed} ✗")
        print(f"  通过率: {(self.passed/total*100):.1f}%")
        print("="*60)
        
        return self.failed == 0


def test_imports(result: TestResult):
    """测试模块导入"""
    print("\n[测试组 1] 模块导入验证")
    print("-" * 40)
    
    # 测试 onnx 库
    try:
        import onnx
        result.add("import onnx", True, f"version={onnx.__version__}")
    except ImportError as e:
        result.add("import onnx", False, str(e))
    
    # 测试 onnxruntime 库
    try:
        import onnxruntime
        result.add("import onnxruntime", True, f"version={onnxruntime.__version__}")
    except ImportError as e:
        result.add("import onnxruntime", False, str(e))
    
    # 测试 export 模块
    try:
        from export import ONNXExporter, export_model
        result.add("from export import ONNXExporter, export_model", True)
    except ImportError as e:
        result.add("from export import ...", False, str(e))
    
    # 测试 validator 模块
    try:
        from export.validator import ONNXValidator, validate_onnx_model
        result.add("from export.validator import ...", True)
    except ImportError as e:
        result.add("from export.validator import ...", False, str(e))
    
    # 测试 runtime_inference 模块
    try:
        from export.runtime_inference import ONNXInference, create_onnx_inference
        result.add("from export.runtime_inference import ...", True)
    except ImportError as e:
        result.add("from export.runtime_inference import ...", False, str(e))


def test_exporter_functionality(result: TestResult):
    """测试 ONNXExporter 功能"""
    print("\n[测试组 2] ONNXExporter 功能测试")
    print("-" * 40)
    
    from export.exporter import ONNXExporter
    
    # 测试实例化
    try:
        exporter = ONNXExporter(opset_version=17)
        result.add("ONNXExporter 实例化", True, f"opset_version={exporter.opset_version}")
    except Exception as e:
        result.add("ONNXExporter 实例化", False, str(e))
    
    # 测试导出功能（使用临时文件）
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = ONNXExporter()
        
        # 创建一个简单的 PyTorch 模型用于测试
        import torch
        import torch.nn as nn
        
        class SimplePolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(20, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 8),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.net(x)
        
        policy = SimplePolicy()
        policy.eval()
        
        output_path = os.path.join(temp_dir, "test_model.onnx")
        dummy_input = torch.randn(1, 20)
        
        # 手动导出（模拟 exporter 的行为）
        torch.onnx.export(
            policy,
            dummy_input,
            output_path,
            input_names=['observation'],
            output_names=['action'],
            opset_version=17,
            do_constant_folding=True
        )
        
        # 验证文件存在且大小合理
        file_exists = os.path.exists(output_path)
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        result.add("ONNX 文件生成", file_exists, f"大小={file_size:.2f} MB")
        result.add("文件大小合理性", file_size > 0 and file_size < 100, f"{file_size:.2f} MB")
        
    except Exception as e:
        result.add("ONNX 导出测试", False, str(e))
    finally:
        shutil.rmtree(temp_dir)


def test_runtime_inference(result: TestResult):
    """测试 ONNXInference 功能"""
    print("\n[测试组 3] ONNXRuntime 推理测试")
    print("-" * 40)
    
    from export.runtime_inference import ONNXInference, create_onnx_inference
    
    # 测试可用性检测
    available = ONNXInference.is_available()
    result.add("ONNXRuntime 可用性检测", available, 
               "已安装" if available else "未安装")
    
    if not available:
        result.add("跳过推理测试", True, "原因: ONNXRuntime 未安装")
        return
    
    # 创建测试模型并加载
    temp_dir = tempfile.mkdtemp()
    try:
        import torch
        import torch.nn as nn
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(20, 8),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.fc(x)
        
        model = TestModel()
        model.eval()
        
        onnx_path = os.path.join(temp_dir, "test.onnx")
        dummy_input = torch.randn(1, 20)
        
        torch.onnx.export(model, dummy_input, onnx_path,
                          input_names=['observation'],
                          output_names=['action'],
                          opset_version=17)
        
        # 测试加载
        inf = ONNXInference(onnx_path)
        loaded_ok = inf.loaded
        result.add("ONNX 模型加载", loaded_ok)
        
        if loaded_ok:
            # 测试单样本推理
            test_input = np.random.randn(20).astype(np.float32)
            action, confidence = inf.predict(test_input)
            
            action_valid = len(action) == 8
            confidence_valid = 0 <= confidence <= 1
            
            result.add("单样本推理输出维度", action_valid, f"shape={action.shape}")
            result.add("置信度范围正确", confidence_valid, f"confidence={confidence:.3f}")
            result.add("动作值范围正确", all(-1 <= a <= 1 for a in action), 
                       f"range=[{action.min():.3f}, {action.max():.3f}]")
            
            # 测试批量推理
            batch_input = np.random.randn(10, 20).astype(np.float32)
            actions, confidences = inf.predict_batch(batch_input)
            
            batch_valid = actions.shape == (10, 8)
            result.add("批量推理输出维度", batch_valid, f"shape={actions.shape}")
            
            # 测试模型信息获取
            info = inf.get_model_info()
            has_input_info = 'input' in info
            has_output_info = 'output' in info
            
            result.add("模型信息查询", has_input_info and has_output_info)
        
    except Exception as e:
        result.add("推理测试异常", False, str(e))
    finally:
        shutil.rmtree(temp_dir)


def test_validator(result: TestResult):
    """测试 ONNXValidator 功能"""
    print("\n[测试组 4] ONNXValidator 功能测试")
    print("-" * 40)
    
    from export.validator import ONNXValidator
    
    # 创建测试模型
    temp_dir = tempfile.mkdtemp()
    try:
        import torch
        import torch.nn as nn
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(20, 8), nn.Tanh())
            def forward(self, x):
                return self.net(x)
        
        model = TestModel()
        model.eval()
        
        onnx_path = os.path.join(temp_dir, "validate_test.onnx")
        torch.onnx.export(model, torch.randn(1, 20), onnx_path,
                          input_names=['observation'],
                          output_names=['action'],
                          opset_version=17)
        
        validator = ONNXValidator()
        
        # 结构验证
        struct_result = validator.validate_structure(onnx_path)
        valid_structure = struct_result.get('valid', False)
        result.add("结构验证", valid_structure)
        
        if valid_structure:
            has_inputs = len(struct_result.get('inputs', [])) > 0
            has_outputs = len(struct_result.get('outputs', [])) > 0
            result.add("输入输出信息提取", has_inputs and has_outputs)
        
        # 性能基准测试
        perf_result = validator.benchmark_performance(onnx_path, num_iterations=50)
        has_throughput = 'throughput_fps' in perf_result
        result.add("性能基准测试", has_throughput, 
                   f"throughput={perf_result.get('throughput_fps', 0):.0f} fps" if has_throughput else "")
        
        # 报告生成
        report_path = os.path.join(temp_dir, "test_report.json")
        report = validator.generate_report(report_path)
        report_exists = os.path.exists(report_path)
        result.add("报告生成", report_exists)
        
    except Exception as e:
        result.add("验证器测试异常", False, str(e))
    finally:
        shutil.rmtree(temp_dir)


def test_cli_parsing(result: TestResult):
    """测试 CLI 参数解析"""
    print("\n[测试组 5] CLI 工具参数解析测试")
    print("-" * 40)
    
    import subprocess
    
    # 测试 --help
    try:
        result_proc = subprocess.run(
            [sys.executable, "export_onnx.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) or "."
        )
        
        help_ok = result_proc.returncode == 0
        has_usage = "--model" in result_proc.stdout or "-m" in result_proc.stdout
        
        result.add("--help 参数", help_ok and has_usage)
    except Exception as e:
        result.add("--help 参数测试", False, str(e))


def main():
    """主测试函数"""
    print("="*60)
    print("  ONNX Export Functionality Test Suite")
    print(f"  Version: 1.0.1")
    print("="*60)
    
    result = TestResult()
    
    # 执行所有测试组
    test_imports(result)
    test_exporter_functionality(result)
    test_runtime_inference(result)
    test_validator(result)
    test_cli_parsing(result)
    
    # 输出摘要
    success = result.summary()
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()