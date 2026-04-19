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
ONNX 模型导出命令行工具

用法示例:
    python export_onnx.py --model models/best_model.zip --output chair_model.onnx
    python export_onnx.py --model models/best_model.zip --dynamic-batch --validate
    python export_onnx.py --validate-only chair_model.onnx
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='将 Stable-Baselines3 PPO 模型导出为 ONNX 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --model best_model.zip --output model.onnx
  %(prog)s --model best_model.zip --dynamic-batch --validate
  %(prog)s --validate-only model.onnx --benchmark
  %(prog)s --model best_model.zip --output model.onnx --report report.json
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Stable-Baselines3 模型路径 (.zip)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出 ONNX 文件路径（默认自动生成）'
    )
    
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        default=False,
        help='使用动态 batch 维度'
    )
    
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX OpSet 版本（默认: 17）'
    )
    
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        default=False,
        help='导出后进行数值一致性验证'
    )
    
    parser.add_argument(
        '--validate-only',
        type=str,
        default=None,
        help='仅验证现有 ONNX 模型（不导出）'
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        default=False,
        help='运行性能基准测试'
    )
    
    parser.add_argument(
        '--report', '-r',
        type=str,
        default=None,
        help='输出验证报告到 JSON 文件'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='显示详细输出'
    )
    
    args = parser.parse_args()
    
    if not args.model and not args.validate_only:
        parser.error("必须指定 --model 或 --validate-only 参数")
        return 1
    
    print("="*60)
    print("  Ergonomic Chair AI Training - ONNX Export Tool")
    print(f"  Version: 1.0.1")
    print("="*60)
    print()
    
    try:
        if args.validate_only:
            # 仅验证模式
            return validate_mode(args)
        else:
            # 导出模式
            return export_mode(args)
            
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        return 1
    except Exception as e:
        logger.error(f"错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def export_mode(args) -> int:
    """执行导出操作"""
    from export.exporter import export_model, ONNXExporter
    from export.validator import ONNXValidator
    
    # 检查模型文件
    if not os.path.exists(args.model):
        raise FileNotFoundError(args.model)
    
    # 自动生成输出路径
    output_path = args.output
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(args.model))[0]
        suffix = "_dynamic" if args.dynamic_batch else ""
        output_path = f"{base_name}{suffix}.onnx"
    
    print(f"[1/3] 加载模型...")
    print(f"      输入: {args.model}")
    print(f"      输出: {output_path}")
    print(f"      动态Batch: {'是' if args.dynamic_batch else '否'}")
    print(f"      OpSet版本: {args.opset}")
    print()
    
    # 执行导出
    exporter = ONNXExporter(opset_version=args.opset)
    export_info = exporter.export_to_onnx(
        model_path=args.model,
        output_path=output_path,
        dynamic_batch=args.dynamic_batch,
        verbose=args.verbose
    )
    
    if not export_info.get("success"):
        print("[ERROR] 导出失败!")
        return 1
    
    print(f"[✓] 导出成功!")
    print(f"      文件大小: {export_info['file_size_mb']:.2f} MB")
    print(f"      架构: {export_info['model_architecture']}")
    print()
    
    # 结构验证
    print(f"[2/3] 验证模型结构...")
    validator = ONNXValidator()
    structure_result = validator.validate_structure(output_path)
    
    if structure_result.get("valid"):
        print(f"[✓] 结构验证通过!")
        print(f"      输入: {structure_result['inputs'][0]['name']} {structure_result['inputs'][0]['shape']}")
        print(f"      输出: {structure_result['outputs'][0]['name']} {structure_result['outputs'][0]['shape']}")
        print(f"      算子数量: {structure_result['op_count']}")
    else:
        print(f"[✗] 结构验证失败: {structure_result.get('error')}")
        return 1
    print()
    
    # 数值一致性验证
    if args.validate:
        print(f"[3/3] 数值一致性验证...")
        numerical_result = validator.validate_numerical_consistency(
            sb3_model_path=args.model,
            onnx_path=output_path
        )
        
        if numerical_result.get("passed"):
            print(f"[✓] 数值验证通过!")
            print(f"      最大误差: {numerical_result['max_error']:.2e}")
            print(f"      平均误差: {numerical_result['mean_error']:.2e}")
        else:
            print(f"[✗] 数值验证失败!")
            print(f"      最大误差: {numerical_result['max_error']:.2e}")
            print(f"      (阈值: {numerical_result['tolerance']:.2e})")
            return 1
    else:
        print(f"[跳过] 数值验证 (--validate 启用)")
    
    print()
    
    # 性能基准测试
    if args.benchmark:
        print(f"运行性能基准测试...")
        perf_result = validator.benchmark_performance(output_path)
        print(f"[✓] 性能测试完成!")
        print(f"      吞吐量: {perf_result['throughput_fps']:.0f} fps")
        print(f"      平均延迟: {perf_result['latency_ms']['mean']:.4f} ms")
        print(f"      P95延迟: {perf_result['latency_ms']['p95']:.4f} ms")
        print(f"      P99延迟: {perf_result['latency_ms']['p99']:.4f} ms")
        print()
    
    # 生成报告
    if args.report:
        report = validator.generate_report(args.report)
        print(f"[✓] 报告已保存: {args.report}")
    
    # 最终摘要
    print("="*60)
    print("  导出完成摘要")
    print("="*60)
    print(f"  模型文件: {output_path}")
    print(f"  文件大小: {export_info['file_size_mb']:.2f} MB")
    print(f"  导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  状态: ✓ 成功")
    print("="*60)
    
    return 0


def validate_mode(args) -> int:
    """仅执行验证"""
    from export.validator import ONNXValidator, validate_onnx_model
    
    if not os.path.exists(args.validate_only):
        raise FileNotFoundError(args.validate_only)
    
    print(f"验证 ONNX 模型: {args.validate_only}")
    print()
    
    results = validate_onnx_model(
        onnx_path=args.validate_only,
        run_benchmark=args.benchmark,
        output_report=args.report
    )
    
    overall = results.get("overall_status", "UNKNOWN")
    
    print("="*60)
    print(f"  验证结果: {'✓ PASS' if overall == 'PASS' else '✗ FAIL'}")
    print("="*60)
    
    if args.verbose:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
