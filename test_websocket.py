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
WebSocket 实时传感器数据接口测试脚本

测试内容:
1. WebSocket 连接建立与断开
2. 传感器数据实时流
3. 心跳机制
4. 多用户并发连接
5. 异常场景处理

用法:
    python test_websocket.py [test_type]
    
    test_type:
        - basic: 基本连接测试（默认）
        - streaming: 高频数据流测试
        - heartbeat: 心跳机制测试
        - multi_user: 多用户并发测试
        - all: 运行所有测试
    
    示例:
        python test_websocket.py              # 基本测试
        python test_websocket.py streaming    # 流式测试
        python test_websocket.py all          # 全部测试
"""

import os
import sys
import asyncio
import json
import time
import logging
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
        print(f"  WebSocket 测试结果汇总")
        print("="*60)
        print(f"  总计: {total} 个测试")
        print(f"  通过: {self.passed} ✓")
        print(f"  失败: {self.failed} ✗")
        print(f"  通过率: {(self.passed/total*100):.1f}%")
        print("="*60)
        
        return self.failed == 0


def generate_sensor_data(
    frame_id: int = 0,
    sitting_duration: float = 0.0,
    fatigue_level: float = 0.0
) -> Dict[str, Any]:
    """
    生成模拟传感器数据
    
    Args:
        frame_id: 帧编号（用于生成变化的数据）
        sitting_duration: 静坐时长（秒）
        fatigue_level: 疲劳程度 (0-1)
        
    Returns:
        传感器数据字典
    """
    import numpy as np
    
    # 模拟压力矩阵（8x8）
    base_pressure = 0.2 + 0.1 * np.sin(frame_id * 0.1)
    pressure_matrix = (
        base_pressure + 
        np.random.uniform(-0.05, 0.05, (8, 8))
    ).tolist()
    
    # 确保中心区域压力更高（模拟坐姿）
    for i in range(3, 6):
        for j in range(3, 6):
            pressure_matrix[i][j] += 0.15
    
    # 归一化到 0-1
    max_val = max(max(row) for row in pressure_matrix) or 1
    pressure_matrix = [[min(v/max_val, 1.0) for v in row] for row in pressure_matrix]
    
    # 模拟姿态角度（带轻微波动）
    posture_angles = [
        15.0 + 5 * np.sin(frame_id * 0.05),   # 头部角度
        -8.0 + 3 * np.cos(frame_id * 0.03),   # 肩部角度
        10.0 + 4 * np.sin(frame_id * 0.07)     # 骨盆角度
    ]
    
    return {
        "type": "sensor_data",
        "payload": {
            "pressure_matrix": pressure_matrix,
            "posture_angles": [round(a, 1) for a in posture_angles],
            "sitting_duration": round(sitting_duration, 1),
            "user_weight": 70.0,
            "user_height": 1.70,
            "fatigue_level": round(min(fatigue_level, 1.0), 2)
        }
    }


async def test_basic_connection(base_url: str, result: TestResult):
    """测试基本连接功能"""
    print("\n[测试组 1] 基本 WebSocket 连接")
    print("-" * 40)
    
    try:
        import websockets
        
        uri = f"{base_url}/ws/sensor"
        
        # 测试连接建立
        async with websockets.connect(uri) as ws:
            # 接收欢迎消息
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            welcome_msg = json.loads(response)
            
            is_connected = welcome_msg.get("type") == "connected"
            has_session = "session_id" in welcome_msg
            
            result.add("连接建立成功", is_connected)
            result.add("收到 session_id", has_session)
            
            if has_session:
                session_id = welcome_msg["session_id"]
                logger.info(f"  Session ID: {session_id}")
                
                # 测试发送消息
                sensor_data = generate_sensor_data()
                await ws.send(json.dumps(sensor_data))
                
                # 接收响应
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                adjustment = json.loads(response)
                
                is_adjustment = adjustment.get("type") == "adjustment"
                has_action = "action" in adjustment.get("payload", {})
                
                result.add("收到调整建议", is_adjustment)
                result.add("包含动作向量", has_action)
                
                if has_action:
                    action = adjustment["payload"]["action"]
                    has_8_dims = len(action) == 8
                    values_in_range = all(-1 <= v <= 1 for v in action.values())
                    
                    result.add("动作维度正确 (8维)", has_8_dims)
                    result.add("动作值范围 [-1,1]", values_in_range)
                    
                    latency = adjustment["payload"].get("processing_latency_ms", 0)
                    is_fast = latency < 100  # 目标 <100ms
                    result.add(f"延迟 <100ms ({latency:.1f}ms)", is_fast)
        
    except ImportError:
        result.add("websockets 库未安装", False, "请运行: pip install websockets")
    except Exception as e:
        result.add("基本连接测试异常", False, str(e))


async def test_heartbeat(base_url: str, result: TestResult):
    """测试心跳机制"""
    print("\n[测试组 2] 心跳保活机制")
    print("-" * 40)
    
    try:
        import websockets
        
        uri = f"{base_url}/ws/sensor"
        
        async with websockets.connect(uri) as ws:
            # 接收欢迎消息
            await ws.recv()
            
            # 发送 ping
            ping_time = time.time()
            ping_msg = json.dumps({"type": "ping"})
            await ws.send(ping_msg)
            
            # 接收 pong
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            pong_msg = json.loads(response)
            
            is_pong = pong_msg.get("type") == "pong"
            round_trip = (time.time() - ping_time) * 1000
            
            result.add("收到 pong 响应", is_pong)
            result.add(f"心跳往返 <500ms ({round_trip:.1f}ms)", round_trip < 500)
            
            # 测试多次心跳
            for i in range(3):
                await ws.send(json.dumps({"type": "ping"}))
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(response)
                
                if msg.get("type") != "pong":
                    result.add(f"心跳 #{i+1} 失败", False)
                    break
            else:
                result.add("连续心跳正常 (3次)", True)
        
    except Exception as e:
        result.add("心跳测试异常", False, str(e))


async def test_streaming(base_url: str, result: TestResult):
    """测试高频数据流"""
    print("\n[测试组 3] 高频数据流传输")
    print("-" * 40)
    
    try:
        import websockets
        
        uri = f"{base_url}/ws/sensor"
        num_frames = 30  # 1秒 @ 30fps
        latencies = []
        
        async with websockets.connect(uri) as ws:
            # 接收欢迎消息
            await ws.recv()
            
            start_time = time.time()
            
            # 发送高频数据流
            for i in range(num_frames):
                frame_start = time.time()
                
                # 生成递增的静坐时间和疲劳度
                sensor_data = generate_sensor_data(
                    frame_id=i,
                    sitting_duration=1800 + i * 10,
                    fatigue_level=min(0.3 + i * 0.01, 0.9)
                )
                
                await ws.send(json.dumps(sensor_data))
                
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    recv_time = time.time()
                    
                    latencies.append((recv_time - frame_start) * 1000)
                    
                    # 解析响应
                    msg = json.loads(response)
                    if msg.get("type") not in ["adjustment", "alert"]:
                        result.add(f"帧 {i+1}: 非预期响应类型", False, msg.get("type"))
                        
                except asyncio.TimeoutError:
                    result.add(f"帧 {i+1}: 响应超时", False)
            
            total_time = time.time() - start_time
            
            # 统计结果
            avg_latency = sum(latencies) / len(latencies) if latencies else float('inf')
            p95_latency = sorted(latencies)[int(len(latencies)*0.95)] if latencies else float('inf')
            fps = num_frames / total_time if total_time > 0 else 0
            
            result.add(f"成功发送 {num_frames}/{num_frames} 帧", len(latencies) == num_frames)
            result.add(f"平均延迟 <100ms ({avg_latency:.1f}ms)", avg_latency < 100)
            result.add(f"P95延迟 <150ms ({p95_latency:.1f}ms)", p95_latency < 150)
            result.add(f"吞吐量 >20fps ({fps:.1f}fps)", fps > 20)
        
    except Exception as e:
        result.add("流式测试异常", False, str(e))


async def test_multi_user(base_url: str, result: TestResult):
    """测试多用户并发连接"""
    print("\n[测试组 4] 多用户并发连接")
    print("-" * 40)
    
    try:
        import websockets
        
        uri = f"{base_url}/ws/sensor"
        num_users = 5
        sessions = {}
        
        async def user_session(user_id: int):
            """单个用户的会话"""
            async with websockets.connect(uri) as ws:
                # 接收欢迎消息并记录 session
                response = await ws.recv()
                welcome = json.loads(response)
                session_id = welcome.get("session_id", "")
                sessions[user_id] = session_id
                
                # 发送一次数据
                sensor_data = generate_sensor_data(frame_id=user_id * 100)
                await ws.send(json.dumps(sensor_data))
                
                # 接收响应
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(response)
                
                return {
                    "user_id": user_id,
                    "session_id": session_id,
                    "success": msg.get("type") == "adjustment",
                    "session_unique": len(set(sessions.values())) == user_id + 1
                }
        
        # 并发启动多个会话
        tasks = [user_session(i) for i in range(num_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        unique_sessions = set()
        
        for r in results:
            if isinstance(r, dict):
                if r["success"]:
                    success_count += 1
                unique_sessions.add(r["session_id"])
        
        result.add(f"{num_users} 个用户全部连接成功", success_count == num_users)
        result.add(f"每个用户获得独立 session ({len(unique_sessions)}个)", 
                   len(unique_sessions) == num_users)
        
    except Exception as e:
        result.add("多用户测试异常", False, str(e))


async def test_error_handling(base_url: str, result: TestResult):
    """测试错误处理"""
    print("\n[测试组 5] 错误处理")
    print("-" * 40)
    
    try:
        import websockets
        
        uri = f"{base_url}/ws/sensor"
        
        async with websockets.connect(uri) as ws:
            await ws.recv()  # 欢迎消息
            
            # 测试发送非法 JSON
            await ws.send("not valid json")
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            error_msg = json.loads(response)
            
            is_error = error_msg.get("type") == "error"
            result.add("非法 JSON 返回错误消息", is_error)
            
            # 测试发送缺少必要字段的消息
            incomplete_msg = json.dumps({
                "type": "sensor_data",
                "payload": {}  # 缺少必要字段
            })
            await ws.send(incomplete_msg)
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            error_msg = json.loads(response)
            
            is_error_or_warning = error_msg.get("type") in ["error", "warning"]
            result.add("不完整数据返回警告/错误", is_error_or_warning)
            
            # 测试未知消息类型
            unknown_msg = json.dumps({"type": "unknown_type"})
            await ws.send(unknown_msg)
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            warning_msg = json.loads(response)
            
            is_warning = warning_msg.get("type") == "warning"
            result.add("未知消息类型返回警告", is_warning)
        
    except Exception as e:
        result.add("错误处理测试异常", False, str(e))


async def run_all_tests(base_url: str):
    """运行所有测试"""
    result = TestResult()
    
    print("="*60)
    print("  WebSocket Real-time Sensor API Test Suite")
    print(f"  Target: {base_url}")
    print("="*60)
    
    # 检查 websockets 库
    try:
        import websockets
        result.add("websockets 库可用", True, f"version={websockets.__version__}")
    except ImportError:
        result.add("websockets 库可用", False, "请安装: pip install websockets")
        result.summary()
        return result.failed == 0
    
    # 执行测试组
    await test_basic_connection(base_url, result)
    await test_heartbeat(base_url, result)
    await test_streaming(base_url, result)
    await test_multi_user(base_url, result)
    await test_error_handling(base_url, result)
    
    return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WebSocket API 测试工具')
    parser.add_argument(
        'test_type',
        nargs='?',
        default='basic',
        choices=['basic', 'streaming', 'heartbeat', 'multi_user', 'all'],
        help='测试类型'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='服务器 URL'
    )
    
    args = parser.parse_args()
    
    base_url = args.url.replace('http://', 'ws://')
    
    if args.test_type == 'all':
        result = asyncio.run(run_all_tests(base_url))
    elif args.test_type == 'basic':
        result = TestResult()
        asyncio.run(test_basic_connection(base_url, result))
        asyncio.run(test_heartbeat(base_url, result))
        result.summary()
    elif args.test_type == 'streaming':
        result = TestResult()
        asyncio.run(test_streaming(base_url, result))
        result.summary()
    elif args.test_type == 'heartbeat':
        result = TestResult()
        asyncio.run(test_heartbeat(base_url, result))
        result.summary()
    elif args.test_type == 'multi_user':
        result = TestResult()
        asyncio.run(test_multi_user(base_url, result))
        result.summary()
    
    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    main()
