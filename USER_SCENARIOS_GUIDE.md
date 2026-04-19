# 🪑 人体工学座椅 AI 系统 - 用户场景功能指南

> **版本**: v2.3.0 | **更新日期**: 2026-04-19 | **作者**: Harley Wang (王华)

本文档从**实际用户使用场景**出发，详细介绍系统各功能模块的协同工作方式，帮助不同角色的用户快速找到适合自己的使用路径。

---

## 📖 目录

1. [系统架构总览](#1-系统架构总览)
2. [场景一：算法研究人员 - 模型训练与优化](#2-场景一算法研究人员---模型训练与优化)
3. [场景二：产品工程师 - 模型部署与集成](#3-场景二产品工程师---模型部署与集成)
4. [场景三：终端用户 - 实时健康监测](#4-场景三终端用户---实时健康监测)
5. [场景四：数据分析师 - 批量数据分析](#5-场景四数据分析师---批量数据分析)
6. [场景五：第三方开发者 - API 集成开发](#6-场景五第三方开发者---api-集成开发)
7. [场景六：运维工程师 - 系统监控与管理](#7-场景六运维工程师---系统监控与管理)
8. [功能矩阵速查表](#8-功能矩阵速查表)

---

## 1. 系统架构总览

### 1.1 核心功能模块

```
┌─────────────────────────────────────────────────────────────────────┐
│                    人体工学座椅 AI 系统 v2.0                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   训练系统     │    │  推理服务     │    │    姿态识别引擎      │   │
│  │ (training/)  │───▶│ (service.py) │◀──│(posture_classifier)  │   │
│  └──────────────┘    └──────┬───────┘    └──────────────────────┘   │
│                             │                                        │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      FastAPI 应用层                           │  │
│  ├─────────────────────┬──────────────────────┬─────────────────┤  │
│  │   REST API 路由      │   WebSocket 路由     │   导出/验证工具   │  │
│  │ (routes/chair.py)   │(routes/websocket.py)│  (export/)       │  │
│  ├─────────────────────┴──────────────────────┴─────────────────┤  │
│  │                     数据模型层 (models.py)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  输入: 压力矩阵 + 姿态角度 + 用户信息                                │
│  输出: 座椅调整建议 + 姿态分析 + 健康提醒                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 技术栈概览

| 层级 | 技术 | 用途 |
|------|------|------|
| **AI/ML** | Stable-Baselines3 (PPO) | 强化学习训练 |
| **物理仿真** | PyBullet | 数字孪生环境 |
| **Web框架** | FastAPI + Uvicorn | 高性能 API 服务 |
| **实时通信** | WebSocket | 低延迟双向通信 |
| **模型格式** | PyTorch / ONNX | 训练 / 部署 |
| **数据验证** | Pydantic | 请求/响应校验 |

---

## 2. 场景一：算法研究人员 - 模型训练与优化

### 👤 用户画像
- **角色**: AI 算法工程师、强化学习研究员
- **目标**: 训练高性能座椅控制策略，优化奖励函数，实验新算法
- **技能要求**: Python、RL 理论、PyTorch

### 🎯 使用流程

```
Step 1: 环境搭建 → Step 2: 配置参数 → Step 3: 启动训练 → Step 4: 监控指标 → Step 5: 评估导出
```

### 📦 功能详解

#### 2.1 训练环境构建 (`training/environment.py`)

**功能描述**: 构建数字孪生训练场，模拟真实座椅-人体交互

```python
from training.environment import ChairEnv

# 创建训练环境（支持自定义人体特征）
env = ChairEnv(
    user_height=1.70,        # 用户身高 (m)
    user_weight=70,          # 用户体重 (kg)
    body_type="average",     # 体型: thin/average/heavy
    enable_fatigue=True,     # 启用肌肉疲劳模型
    sensor_noise=0.05        # 传感器噪声水平
)

# 观察空间: 20维向量
# - 压力特征 (8维): mean, std, max, percentile_90, high_pressure_ratio
# - 角度特征 (3维): head, shoulder, pelvis (归一化)
# - 时间特征 (1维): sitting_duration (归一化)
# - 用户特征 (2维): weight, height (归一化)
# - 疲劳度 (1维): fatigue_level
# - 其他 (5维): max_pressure, pressure_metrics...

# 动作空间: 8维连续动作 (-1 到 1)
# - seat_height: 座垫高度调整
# - backrest_angle: 靠背角度调整
# - lumbar_position: 腰托位置调整
# - lumbar_thickness: 腰托厚度调整
# - headrest_position: 头枕位置调整
# - headrest_angle: 头枕角度调整
# - left_armrest: 左扶手高度
# - right_armrest: 右扶手高度
```

#### 2.2 奖励函数设计 (`training/rewards.py`)

**功能描述**: 多目标优化奖励函数，平衡舒适度与健康

```python
# 奖励函数公式:
R_total = w₁×R_舒适 - w₂×R_压力 - w₃×R_静态 - w₄×R_能耗

# 各子奖励:
# R_舒适: 姿态角度偏离理想值的程度 (权重 40%)
# R_压力: 压力分布均匀性，避免局部高压 (权重 30%)
# R_静态: 鼓励动态调整，避免长时间固定姿势 (权重 20%)
# R_能耗: 控制调整幅度，避免频繁大幅变动 (权重 10%)
```

**可调参数**:
```python
reward_config = {
    "weights": {
        "comfort": 0.4,
        "pressure": 0.3,
        "static_penalty": 0.2,
        "energy": 0.1
    },
    "angle_thresholds": {
        "head_ideal": 5,       # 头部理想角度 (°)
        "shoulder_ideal": 5,   # 肩部理想角度 (°)
        "pelvis_ideal": 5      # 骨盆理想角度 (°)
    },
    "pressure_thresholds": {
        "max_allowed": 0.85,   # 最大允许压力
        "imbalance_limit": 0.25 # 左右不平衡阈值
    }
}
```

#### 2.3 PPO 训练循环 (`training/train.py`)

**启动训练**:

```bash
# 方式1: 直接运行
python -m training.train \
    --total-timesteps 1000000 \
    --learning-rate 3e-4 \
    --batch-size 64 \
    --n-steps 2048 \
    --save-dir ./models

# 方式2: 通过 REST API 远程训练
curl -X POST http://localhost:8000/api/v1/chair/train \
  -H "Content-Type: application/json" \
  -d '{
    "total_timesteps": 500000,
    "learning_rate": 0.0003,
    "batch_size": 64,
    "save_dir": "./models"
  }'
```

**训练监控**:
```python
# TensorBoard 可视化 (如果已安装)
tensorboard --logdir ./logs/

# 监控指标:
# - episode_reward: 每回合累积奖励
# - episode_length: 回合长度
# - policy_loss: 策略网络损失
# - value_loss: 价值网络损失
# - entropy: 策略熵 (探索程度)
```

#### 2.4 动态人体特征 (`training/human_model.py`)

**功能**: 模拟真实用户的动态变化，提升模型泛化能力

```python
class DynamicHumanModel:
    """
    支持的动态特征:
    
    1. 肌肉疲劳模型 (Muscle Fatigue Model)
       - 长时间维持同一姿势 → 疲劳累积
       - 影响姿态保持能力 → 增加角度抖动
       
    2. 重心偏移 (Center of Mass Shift)
       - 疲劳导致重心逐渐前移/侧移
       - 影响压力分布模式
       
    3. 体型分布 (Body Type Distribution)
       - thin: 偏瘦用户，压力集中
       - average: 标准用户
       - heavy: 偏重用户，压力大
    """
    
    def update_fatigue(self, current_posture, duration):
        """更新疲劳状态"""
        
    def shift_com(self, fatigue_level):
        """计算重心偏移"""
        
    def randomize_user_features(self):
        """随机化用户特征以增强鲁棒性"""
```

#### 2.5 模型评估与对比

```python
from training.evaluation import evaluate_policy

# 评估训练好的策略
results = evaluate_policy(
    model_path="./models/best_model.zip",
    n_episodes=100,
    render=False  # 设为 True 可可视化
)

print(f"平均奖励: {results['mean_reward']:.2f}")
print(f"舒适度评分: {results['avg_comfort']:.2f}")
print(f"压力风险: {results['pressure_risk']}")
print(f"姿态识别准确率: {results['posture_accuracy']:.2%}")
```

### 🔬 高级用法

#### 自定义奖励函数

```python
from training.rewards import RewardFunction

class CustomReward(RewardFunction):
    def __init__(self):
        super().__init__()
        # 添加新的奖励维度
        self.add_reward_component(
            name="posture_change_penalty",
            weight=0.15,
            function=self._calculate_change_penalty
        )
    
    def _calculate_change_penalty(self, state, action):
        """惩罚剧烈的姿态变化"""
        angle_diff = np.abs(state['posture_angles'] - self.prev_angles)
        return -np.mean(angle_diff)
```

#### 多模型对比实验

```python
import json
from stable_baselines3 import PPO

models_to_compare = [
    "./models/v1_baseline.zip",
    "./models/v2_with_fatigue.zip",
    "./models/v3_custom_reward.zip"
]

comparison_results = {}

for model_path in models_to_compare:
    model = PPO.load(model_path)
    results = self.evaluate(model, n_episodes=50)
    comparison_results[model_path] = results
    
with open("experiment_results.json", "w") as f:
    json.dump(comparison_results, f, indent=2)
```

### ✅ 场景一检查清单

- [ ] 环境配置完成（PyBullet、依赖安装）
- [ ] 训练超参数调优（lr、batch_size、n_steps）
- [ ] 奖励函数定制化
- [ ] 动态人体特征启用
- [ ] TensorBoard 监控配置
- [ ] 模型评估指标达标
- [ ] 最佳模型保存和版本管理

---

## 3. 场景二：产品工程师 - 模型部署与集成

### 👤 用户画像
- **角色**: 嵌入式工程师、MLOps 工程师、硬件集成工程师
- **目标**: 将训练好的模型部署到生产环境或边缘设备
- **技能要求**: Linux、Docker、ONNX、嵌入式系统

### 🎯 使用流程

```
Step 1: 模型选择 → Step 2: 格式转换 → Step 3: 性能优化 → Step 4: 部署测试 → Step 5: 监控维护
```

### 📦 功能详解

#### 3.1 ONNX 模型导出 (`export/exporter.py`)

**功能**: 将 PyTorch/PPO 模型转换为跨平台 ONNX 格式

```python
from export.exporter import export_model

# 导出为 ONNX 格式
result = export_model(
    model_path="./models/trained_ppo_model.zip",  # SB3 模型
    output_path="./exported_models/chair_ai.onnx",
    dynamic_batch=True,  # 支持动态 batch size
    opset_version=17    # ONNX 算子集版本
)

print(result)
# {
#   "success": True,
#   "format": "onnx",
#   "output_path": "./exported_models/chair_ai.onnx",
#   "file_size_mb": 15.8,
#   "input_shape": [1, 20],
#   "output_shape": [1, 8],
#   "opset_version": 17
# }
```

**通过 REST API 导出**:

```bash
# HTTP 请求
POST /api/v1/model/export

{
  "model_path": "./models/best_model.zip",
  "output_format": "onnx",
  "dynamic_batch": true,
  "optimize": true
}

# 响应
{
  "success": true,
  "message": "Model exported successfully",
  "format": "onnx",
  "output_path": "/exports/chair_v2.0.onnx",
  "file_size_mb": 12.3,
  "export_info": {
    "input_spec": {"shape": [1, 20], "dtype": "float32"},
    "output_spec": {"shape": [1, 8], "dtype": "float32"},
    "optimization_applied": ["constant_folding", "operator_fusion"]
  }
}
```

#### 3.2 模型验证工具 (`export/validator.py`)

**功能**: 验证导出的 ONNX 模型正确性和性能

```python
from export.validator import ONNXValidator

validator = ONNXValidator()

# 验证模型结构
validation_result = validator.validate(
    model_path="./exported_models/chair_ai.onnx",
    test_input_dim=20,
    test_output_dim=8
)

print(validation_result.is_valid)  # True/False
print(validation_result.errors)    # 错误列表
print(validation_result.warnings)  # 警告列表

# 性能基准测试
benchmark = validator.benchmark(
    model_path="./exported_models/chair_ai.onnx",
    n_runs=1000,
    warmup=100
)

print(f"平均推理延迟: {benchmark.avg_latency_ms:.2f} ms")
print(f"P99 延迟: {benchmark.p99_latency_ms:.2f} ms")
print(f"吞吐量: {benchmark.throughput_fps:.0f} FPS")
```

#### 3.3 ONNX Runtime 推理 (`export/runtime_inference.py`)

**功能**: 轻量级推理引擎，适合边缘设备部署

```python
from export.runtime_inference import ONNXInference

# 初始化推理引擎
inference = ONNXInference(
    model_path="./exported_models/chair_ai.onnx",
    provider="CPUExecutionProvider"  # 或 CUDAExecutionProvider, CoreMLExecutionProvider
)

# 单次推理
sensor_data = {
    "pressure_matrix": [[...]],  # 8x8
    "posture_angles": [15, -8, 10],
    "sitting_duration": 1800,
    "user_weight": 70,
    "user_height": 1.70,
    "fatigue_level": 0.4
}

action_vector, confidence = inference.predict(sensor_data)
print(f"调整动作: {action_vector}")
print(f"置信度: {confidence:.2%}")

# 批量推理 (高吞吐量场景)
batch_data = [sensor_data_1, sensor_data_2, ..., sensor_data_n]
actions, confidences = inference.predict_batch(batch_data)
```

#### 3.4 服务端加载 ONNX 模型

```python
from api.service import ChairAIService

service = ChairAIService()

# 加载 ONNX 模型用于推理
success = service.load_onnx_model("./exported_models/chair_ai.onnx")

if success:
    print("ONNX 模型加载成功")
    print(f"模型路径: {service.onnx_inf.model_path}")
    print(f"推理后端: {service.onnx_inf.provider}")
else:
    print("ONNX 模型加载失败，将使用规则回退")
```

### 🚀 部署方案

#### 方案 A: 云端部署 (推荐用于大规模服务)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  chair-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CHAIR_MODEL_PATH=/models/chair_ai.onnx
    volumes:
      - ./models:/models
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'

  # 可选: Redis 用于 WebSocket 会话管理
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

#### 方案 B: 边缘部署 (树莓派/嵌入式设备)

**硬件要求**:
- CPU: ARM Cortex-A72 (树莓派 4) 或更高
- 内存: ≥ 2GB RAM
- 存储: ≥ 16GB SD卡

**优化步骤**:

```bash
# 1. 安装轻量级运行时
pip install onnxruntime-arm64  # ARM 优化版

# 2. 使用量化模型减小体积
python -m export.quantize \
    --input ./chair_ai.onnx \
    --output ./chair_ai_int8.onnx \
    --mode int8

# 3. 启动优化的服务
CHAIR_MODEL_PATH=./chair_ai_int8.onnx \
API_HOST=0.0.0.0 \
API_PORT=8080 \
python -m api.main
```

**性能预期** (树莓派 4):
```
推理延迟: ~15-25ms (INT8 量化)
吞吐量: ~40-60 FPS
内存占用: ~150MB
```

#### 方案 C: 移动端部署 (iOS/Android)

```python
# 使用 CoreML (iOS) 或 NNAPI (Android)
from export.mobile_export import export_to_coreml, export_to_nnapi

# iOS
export_to_coreml(
    onnx_path="./chair_ai.onnx",
    output_path="./ChairAI.mlmodel"
)

# Android
export_to_nnapi(
    onnx_path="./chair_ai.onnx",
    output_path="./chair_ai.tflite"
)
```

### ✅ 场景二检查清单

- [ ] 模型格式转换 (PyTorch → ONNX)
- [ ] 模型验证和正确性测试
- [ ] 性能基准测试达标
- [ ] 目标平台适配 (CPU/GPU/NPU)
- [ ] Docker 容器化打包
- [ ] 生产环境配置 (环境变量、资源限制)
- [ ] 监控和日志收集
- [ ] 模型版本管理和灰度发布

---

## 4. 场景三：终端用户 - 实时健康监测

### 👤 用户画像
- **角色**: 办公室职员、远程工作者、长时间伏案人员
- **目标**: 获得实时的坐姿纠正建议，改善办公健康
- **技能要求**: 无需技术背景，通过 App/Web 界面交互

### 🎯 使用流程

```
打开应用 → 连接座椅传感器 → 实时监测 → 接收提醒 → 查看报告
```

### 📦 功能详解

#### 4.1 REST API 快速查询 (单次检测)

**适用场景**: 手动触发检测、定期快照分析

```bash
# 发送一次传感器数据获取调整建议
curl -X POST http://localhost:8000/api/v1/chair/adjust \
  -H "Content-Type: application/json" \
  -d '{
    "pressure_matrix": [
      [0.1, 0.15, 0.25, 0.35, 0.32, 0.28, 0.18, 0.12],
      [0.12, 0.18, 0.30, 0.45, 0.42, 0.35, 0.20, 0.14],
      ...
    ],
    "posture_angles": [22.0, -12.0, 15.0],
    "sitting_duration": 2700.0,
    "user_weight": 72.0,
    "user_height": 1.68,
    "fatigue_level": 0.55
  }'
```

**响应示例**:

```json
{
  "success": true,
  "timestamp": "2026-04-19T14:30:00.123456",
  "action": {
    "seat_height": 0.1234,
    "backrest_angle": -0.1567,
    "lumbar_position": 0.0891,
    "lumbar_thickness": 0.2345,
    "headrest_position": 0.0678,
    "headrest_angle": -0.0543,
    "left_armrest": -0.0234,
    "right_armrest": 0.0345
  },
  "confidence": 0.8234,
  "comfort_score": 72.5,
  "pressure_risk": "medium",
  "posture_analysis": {
    "head_posture": "slight",
    "shoulder_posture": "normal",
    "pelvis_posture": "slight",
    "sitting_duration_minutes": 45,
    "fatigue_percentage": 55.0,
    "max_pressure_point": 0.789,
    "pressure_balance_score": 0.856,
    "overall_risk_level": "medium",
    "issues_detected": 3,
    "comfort_score": 72.5
  },
  "posture_detail": {
    "posture_type": "forward_lean",
    "posture_name_cn": "前倾/探头",
    "severity": "warning",
    "confidence": 0.823,
    "risk_areas": ["颈椎 (C4-C7)", "肩胛提肌", "斜方肌"],
    "recommended_exercises": [
      "收下巴练习(每次5秒×10次)",
      "胸椎伸展运动",
      "肩胛骨回缩练习"
    ],
    "primary_adjustments": {
      "seat_height": 0.1,
      "backrest_angle": -0.15,
      "headrest_position": 0.2
    },
    "message": "检测到前倾/探头姿势，建议：①将座垫后移 2-3cm ②调直靠背至90-100° ③显示器抬高至视线水平"
  },
  "recommendations": [
    "检测到前倾/探头姿势，建议：①将座垫后移 2-3cm ②调直靠背至90-100° ③显示器抬高至视线水平",
    "静坐时间超过45分钟，建议适当休息",
    "出现轻度疲劳迹象"
  ]
}
```

#### 4.2 WebSocket 实时流 (持续监测) ⭐

**适用场景**: 实时坐姿监测、自动调整、长期健康追踪

**连接建立**:

```javascript
// JavaScript 客户端示例
const ws = new WebSocket("ws://localhost:8000/ws/sensor");

ws.onopen = () => {
    console.log("WebSocket 已连接");
    
    // 发送心跳保持连接
    setInterval(() => {
        ws.send(JSON.stringify({
            type: "ping",
            timestamp: new Date().toISOString()
        }));
    }, 30000); // 每30秒发送一次心跳
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case "connected":
            console.log("会话ID:", data.session_id);
            console.log("支持的消息类型:", data.payload.supported_message_types);
            break;
            
        case "adjustment":
            handleAdjustment(data.payload);
            break;
            
        case "alert":
            showAlertNotification(data.payload);
            break;
            
        case "pong":
            updateLatency(Date.now() - data.payload.server_time);
            break;
    }
};
```

**实时数据发送** (假设传感器频率 30Hz):

```javascript
function sendSensorData(pressureMatrix, postureAngles) {
    const message = {
        type: "sensor_data",
        timestamp: new Date().toISOString(),
        session_id: currentSessionId,
        payload: {
            pressure_matrix: pressureMatrix,     // 8x8 数组
            posture_angles: postureAngles,         // [head, shoulder, pelvis]
            sitting_duration: getSittingDuration(), // 秒
            user_weight: currentUser.weight,
            user_height: currentUser.height,
            fatigue_level: calculateFatigue()      // 0-1
        }
    };
    
    ws.send(JSON.stringify(message));
}

// 每33ms发送一次 (30Hz)
setInterval(() => {
    const data = readSensors();
    sendSensorData(data.pressure, data.angles);
}, 33);
```

**接收实时响应**:

```javascript
function handleAdjustment(payload) {
    console.log(`处理延迟: ${payload.processing_latency_ms}ms`);
    console.log(`置信度: ${(payload.confidence * 100).toFixed(1)}%`);
    console.log(`舒适度评分: ${payload.comfort_score}`);
    
    // 应用座椅调整
    if (payload.confidence > 0.7) {
        adjustSeat(payload.action);
    }
    
    // 显示姿态详情
    if (payload.posture_detail) {
        displayPostureInfo(payload.posture_detail);
    }
}

function showAlertNotification(alert) {
    switch(alert.alert_type) {
        case "sitting_too_long":
            showNotification("⏰ 已连续静坐太久！", alert.severity);
            playAlertSound();
            break;
            
        case "fatigue_high":
            showNotification("😴 检测到高度疲劳", alert.severity);
            suggestBreak();
            break;
            
        case "posture_warning":
            showNotification("🪑 不良坐姿警告", alert.severity);
            highlightBadPosture();
            break;
    }
}
```

#### 4.3 姿态类型识别详解

**支持的 8 种坐姿类型**:

| # | 姿态 | 英文标识 | 典型表现 | 健康影响 | 系统建议 |
|---|------|----------|----------|----------|----------|
| 1 | **正常坐姿** | `normal` | 所有角度在正常范围 | ✅ 理想 | 保持当前设置 |
| 2 | **前倾/探头** | `forward_lean` | 头部>15°+肩部>10°前倾 | ⚠️ 颈椎压力 | 后移座垫、调直靠背 |
| 3 | **后仰/瘫坐** | `backward_recline` | 靠背角>20°后仰 | ⚠️ 腰椎悬空 | 增强腰托支撑 |
| 4 | **左偏/右偏** | `lateral_tilt` | 骨盆侧倾>12° | ⚠️ 脊柱侧弯 | 平衡扶手高度 |
| 5 | **交叉腿坐** | `crossed_legs` | 压力不对称 | ⚠️ 骨盆旋转 | 双脚平放地面 |
| 6 | **跷二郎腿** | `leg_crossed` | 单侧承重>60% | ⚠⚠ 静脉受压 | ⚠️ 立即放下翘起的腿 |
| 7 | **盘腿坐** | `lotus_position` | 压力分散在外缘 | ⚠ 髋关节压力 | 每30分钟切换姿势 |
| 8 | **前伸坐姿** | `forward_reach` | 身体远离靠背 | ⚠⚠ 腰部无支撑 | 🔴 向后靠近靠背 |

**严重程度分级**:

| 等级 | 标识 | 含义 | UI展示 | 系统行为 |
|------|------|------|--------|----------|
| **IDEAL** | `ideal` | 完美坐姿 | 🟢 绿色 | 无需干预 |
| **GOOD** | `good` | 轻微偏差 | 🟡 黄色提示 | 静默记录 |
| **WARNING** | `warning` | 明显不良 | 🟠 橙色警告 | 弹窗提醒 |
| **DANGER** | `danger` | 严重问题 | 🔴 红色告警 | 强制干预+声音 |

#### 4.4 快速接口 (简化输入)

**适用场景**: 最小化输入，快速获取建议

```bash
# 仅需3个参数即可获得基本建议
GET /api/v1/chair/quick-adjust?sitting_minutes=60&discomfort_level=0.6&body_type=average

# 响应
{
    "success": true,
    "timestamp": "...",
    "action": {...},
    "confidence": 0.75,
    "comfort_score": 65.2,
    "pressure_risk": "medium",
    "posture_analysis": {...},
    "recommendations": ["建议增加腰部支撑"],
    "posture_detail": {...}
}
```

#### 4.5 演示接口 (无需传感器)

**适用场景**: 产品演示、功能体验、开发调试

```bash
# 获取预设的示例数据和完整响应
GET /api/v1/chair/demo
```

### 📱 终端用户体验设计建议

#### 移动端 App 界面布局

```
┌────────────────────────────────────┐
│  🪑 智能座椅助手          [⚙️设置] │
├────────────────────────────────────┤
│                                    │
│     ┌──────────────────┐          │
│     │                  │          │
│     │   3D 座椅模型    │          │
│     │   (实时显示调整)  │          │
│     │                  │          │
│     └──────────────────┘          │
│                                    │
│  当前姿态: 前倾/探头  ⚠️ WARNING   │
│                                    │
│  ┌──────────────────────────────┐  │
│  │ 舒适度: ████████░░ 72分      │  │
│  │ 静坐时长: 45分钟             │  │
│  │ 疲劳度: 55%                  │  │
│  └──────────────────────────────┘  │
│                                    │
│  💡 建议:                          │
│  • 将座垫后移 2-3cm               │
│  • 调直靠背至90-100°              │
│  • 显示器抬高至视线水平           │
│                                    │
│  [📊 详细报告]  [🏃 起身活动]      │
│                                    │
└────────────────────────────────────┘
```

#### 桌面端 Dashboard

```
┌──────────────────────────────────────────────────────────────┐
│  人体工学座椅智能监控系统                              v2.0.0  │
├──────────────┬──────────────┬──────────────┬────────────────┤
│              │              │              │                │
│  📈 实时曲线  │  🪑 姿态热力图│  📊 今日统计  │  ⚠️ 健康警报   │
│              │              │              │                │
│  [压力分布]   │  [8x8网格]   │  总时长: 6.5h │  🔴 14:23     │
│  [角度变化]   │  颜色编码:   │  平均舒适度:   │  前伸坐姿>5min │
│  [疲劳趋势]   │  🔴高 🟢低   │  78分         │                │
│              │              │  姿态分布:     │  🟡 13:45     │
│              │              │  正常: 62%     │  疲劳度>70%    │
│              │              │  前倾: 23%     │                │
│              │              │  后仰: 10%     │  🟢 12:10     │
│              │              │  其他: 5%      │  建议休息       │
│              │              │              │                │
├──────────────┴──────────────┴──────────────┴────────────────┤
│  [导出日报]  [历史对比]  [个性化设置]  [专家咨询]  [帮助]     │
└──────────────────────────────────────────────────────────────┘
```

### ✅ 场景三检查清单

- [ ] 传感器硬件连接正常
- [ ] WiFi/蓝牙通信稳定
- [ ] 实时延迟 < 100ms (WebSocket)
- [ ] 告警通知及时送达
- [ ] 历史数据存储和查看
- [ ] 隐私保护设置
- [ ] 离线模式支持 (可选)
- [ ] 多设备同步 (可选)

---

## 5. 场景四：数据分析师 - 批量数据分析

### 👤 用户画像
- **角色**: 健康数据分析师、人因工程研究员、企业健康管理师
- **目标**: 分析大量坐姿数据，发现健康趋势，生成统计报告
- **技能要求**: Python、数据分析、统计学

### 🎯 使用流程

```
数据收集 → 数据清洗 → 批量推理 → 统计分析 → 报告生成
```

### 📦 功能详解

#### 5.1 批量处理接口

**适用场景**: 离线分析、历史数据回溯、批量打标签

```python
import requests
import json
from datetime import datetime, timedelta

# 准备批量数据 (最多100个样本)
batch_data = {
    "samples": []
}

# 生成一天的数据点 (每分钟一个采样点)
base_time = datetime.now() - timedelta(hours=8)
for i in range(480):  # 8小时工作日
    timestamp = base_time + timedelta(minutes=i)
    
    sample = {
        "pressure_matrix": generate_realistic_pressure(i),  # 模拟数据
        "posture_angles": [simulate_head_angle(i), 
                          simulate_shoulder_angle(i),
                          simulate_pelvis_angle(i)],
        "sitting_duration": i * 60,
        "user_weight": 72.0,
        "user_height": 1.68,
        "fatigue_level": min(i / 480, 1.0)  # 疲劳随时间累积
    }
    batch_data["samples"].append(sample)

# 发送批量请求
response = requests.post(
    "http://localhost:8000/api/v1/chair/batch-adjust",
    json=batch_data
)

result = response.json()

print(f"处理样本数: {result['total_samples']}")
print(f"总耗时: {result['processing_time_ms']:.2f} ms")
print(f"平均每样本: {result['processing_time_ms']/len(batch_data['samples']):.2f} ms")
```

**响应示例**:

```json
{
  "success": true,
  "total_samples": 480,
  "processing_time_ms": 1523.45,
  "results": [
    {
      "success": true,
      "timestamp": "2026-04-19T09:00:00",
      "action": {...},
      "confidence": 0.82,
      "comfort_score": 85.3,
      "pressure_risk": "low",
      "posture_analysis": {...},
      "posture_detail": {
        "posture_type": "normal",
        "severity": "ideal",
        ...
      },
      "recommendations": ["姿态良好"]
    },
    // ... 更多结果
  ]
}
```

#### 5.2 数据分析与可视化

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# 将结果转换为 DataFrame
df = pd.DataFrame([r for r in result['results'] if r['success']])

# 提取姿态类型序列
posture_types = df['posture_detail'].apply(lambda x: x['posture_type'])
severity_levels = df['posture_detail'].apply(lambda x: x['severity'])

# ======== 分析 1: 姿态时间分布 ========
posture_counts = Counter(posture_types)

plt.figure(figsize=(12, 6))
plt.bar(posture_counts.keys(), posture_counts.values())
plt.title('今日坐姿类型分布')
plt.xlabel('姿态类型')
plt.ylabel('持续时间占比 (%)')
plt.xticks(rotation=45)
plt.savefig('posture_distribution.png')
plt.show()

# ======== 分析 2: 舒适度变化趋势 ========
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['comfort_score'], label='舒适度', alpha=0.7)
plt.axhline(y=80, color='g', linestyle='--', label='良好线')
plt.axhline(y=60, color='orange', linestyle='--', label='警戒线')
plt.title('舒适度变化趋势 (8小时)')
plt.xlabel('时间 (分钟)')
plt.ylabel('舒适度评分')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('comfort_trend.png')
plt.show()

# ======== 分析 3: 危险时段识别 ========
danger_periods = df[severity_levels == 'danger']
if not danger_periods.empty:
    print("⚠️ 发现危险时段:")
    for idx, row in danger_periods.iterrows():
        print(f"  - {row['timestamp']}: {row['posture_detail']['posture_name_cn']}")

# ======== 分析 4: 疲劳累积分析 ========
fatigue_by_hour = df.groupby(df.index // 60)['posture_analysis'].apply(
    lambda x: np.mean([p['fatigue_percentage'] for p in x])
)

plt.figure(figsize=(10, 5))
plt.plot(range(len(fatigue_by_hour)), fatigue_by_hour.values, 'r-o')
plt.title('疲劳度累积曲线')
plt.xlabel('工作时间 (小时)')
plt.ylabel('疲劳度 (%)')
plt.fill_between(range(len(fatigue_by_hour)), fatigue_by_hour.values, alpha=0.3)
plt.savefig('fatigue_curve.png')
plt.show()
```

#### 5.3 健康风险评估报告生成

```python
def generate_health_report(batch_result):
    """生成综合健康评估报告"""
    
    df = pd.DataFrame([r for r in batch_result['results'] if r['success']])
    
    report = {
        "report_date": datetime.now().isoformat(),
        "analysis_period": f"{df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']}",
        "total_samples": len(df),
        
        # ======== 基础统计 ========
        "statistics": {
            "total_sitting_hours": len(df) / 60,
            "avg_comfort_score": float(df['comfort_score'].mean()),
            "min_comfort_score": float(df['comfort_score'].min()),
            "max_comfort_score": float(df['comfort_score'].max()),
            "std_comfort_score": float(df['comfort_score'].std()),
            
            "avg_confidence": float(df['confidence'].mean()),
            "peak_fatigue": float(max(
                p['fatigue_percentage'] 
                for p in df['posture_analysis']
            ))
        },
        
        # ======== 姿态分析 ========
        "posture_breakdown": dict(Counter(
            df['posture_detail'].apply(lambda x: x['posture_name_cn'])
        )),
        
        "severity_summary": dict(Counter(
            df['posture_detail'].apply(lambda x: x['severity'])
        )),
        
        # ======== 风险评估 ========
        "risk_assessment": {
            "posture_risk_score": calculate_posture_risk(df),
            "fatigue_risk_score": calculate_fatigue_risk(df),
            "pressure_risk_score": calculate_pressure_risk(df),
            "overall_health_score": calculate_overall_health(df)
        },
        
        # ======== 关键发现 ========
        "key_findings": extract_key_findings(df),
        
        # ======== 改进建议 ========
        "recommendations": generate_recommendations(df)
    }
    
    return report

def calculate_posture_risk(df):
    """计算姿态风险分数 (0-100, 越低越好)"""
    severity_weights = {'ideal': 0, 'good': 1, 'warning': 3, 'danger': 10}
    
    risk_sum = sum(
        severity_weights[s] * count 
        for s, count in Counter(
            df['posture_detail'].apply(lambda x: x['severity'])
        ).items()
    )
    
    max_possible_risk = len(df) * 10
    return round((1 - risk_sum / max_possible_risk) * 100, 1)

# 生成并保存报告
report = generate_health_report(result)

with open("health_report_2026-04-19.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
```

**报告示例输出**:

```json
{
  "report_date": "2026-04-19T18:00:00",
  "analysis_period": "09:00:00 ~ 17:00:00 (8小时)",
  "total_samples": 480,
  
  "statistics": {
    "total_sitting_hours": 8.0,
    "avg_comfort_score": 73.2,
    "min_comfort_score": 42.5,
    "max_comfort_score": 95.8,
    "std_comfort_score": 12.3,
    "avg_confidence": 0.81,
    "peak_fatigue": 87.0
  },
  
  "posture_breakdown": {
    "正常坐姿": 298,
    "前倾/探头": 112,
    "后仰/瘫坐": 38,
    "交叉腿坐": 22,
    "跷二郎腿": 8,
    "前伸坐姿": 2
  },
  
  "severity_summary": {
    "ideal": 198,
    "good": 175,
    "warning": 95,
    "danger": 12
  },
  
  "risk_assessment": {
    "posture_risk_score": 72.5,
    "fatigue_risk_score": 58.3,
    "pressure_risk_score": 81.2,
    "overall_health_score": 68.7
  },
  
  "key_findings": [
    "⚠️ 上午10:00-11:00 出现持续前倾姿势 (可能因专注工作)",
    "🔴 下午14:15-14:32 出现前伸坐姿 (午餐后疲劳期)",
    "⚠️ 累计12次跷二郎腿 (右侧为主)",
    "✅ 整体压力分布较为均衡",
    "⚠️ 疲劳度在下午3点达到峰值 (87%)"
  ],
  
  "recommendations": [
    "💺 建议: 调整显示器高度和距离，减少前倾",
    "⏰ 建议: 设置每小时5分钟休息提醒",
    "🏃 建议: 午餐后进行10分钟轻度活动",
    "📱 建议: 开启实时姿态提醒功能",
    "👨‍⚕️ 建议: 如持续不适，请咨询人因工程专家"
  ]
}
```

#### 5.4 多用户对比分析

```python
def compare_multiple_users(user_reports):
    """多用户健康数据对比"""
    
    comparison_data = []
    for user_id, report in user_reports.items():
        comparison_data.append({
            "user_id": user_id,
            "health_score": report["risk_assessment"]["overall_health_score"],
            "avg_comfort": report["statistics"]["avg_comfort_score"],
            "posture_risk": report["risk_assessment"]["posture_risk_score"],
            "fatigue_risk": report["risk_assessment"]["fatigue_risk_score"],
            "danger_count": report["severity_summary"].get("danger", 0),
            "dominant_bad_posture": max(
                report["posture_breakdown"].items(),
                key=lambda x: x[1]
            )[0] if any(k not in ["正常坐姿"] for k in report["posture_breakdown"]) else "N/A"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 排名
    comparison_df['health_rank'] = comparison_df['health_score'].rank(ascending=False)
    
    return comparison_df.sort_values('health_rank')

# 示例: 对比团队成员
team_reports = {
    "user_A": load_report("reports/user_A.json"),
    "user_B": load_report("reports/user_B.json"),
    "user_C": load_report("reports/user_C.json")
}

ranking = compare_multiple_users(team_reports)
print(ranking.to_string(index=False))

# 输出:
# user_id  health_score  avg_comfort  posture_risk  fatigue_risk  danger_count  dominant_bad_posture  health_rank
# user_B          78.3         82.1          85.2          72.1             5                 前倾/探头          1
# user_A          68.7         73.2          72.5          58.3            12               前伸坐姿           2
# user_C          55.4         61.8          58.9          45.2            22               跷二郎腿           3
```

### ✅ 场景四检查清单

- [ ] 数据采集完整性 (无缺失、无异常值)
- [ ] 时间戳同步准确性
- [ ] 批量处理性能满足需求
- [ ] 统计分析方法科学性
- [ ] 可视化图表清晰易懂
- [ ] 报告内容全面且有洞察
- [ ] 隐私保护和数据脱敏
- [ ] 历史数据存储和检索效率

---

## 6. 场景五：第三方开发者 - API 集成开发

### 👤 用户画像
- **角色**: 应用开发者、系统集成商、硬件厂商
- **目标**: 将座椅 AI 能力集成到自己的产品中
- **技能要求**: REST API、WebSocket、任意编程语言

### 🎯 使用流程

```
阅读文档 → 获取 API Key → 开发集成 → 测试调试 → 上线发布
```

### 📦 功能详解

#### 6.1 API 概览

**基础信息**:
```
Base URL: http://localhost:8000
API Version: v1
协议: HTTP/HTTPS + WebSocket
认证: 暂无 (未来可添加 API Key/OAuth)
```

**所有端点列表**:

| 方法 | 路径 | 描述 | 认证 |
|------|------|------|------|
| GET | `/` | API 信息和端点列表 | 无 |
| GET | `/docs` | Swagger 交互式文档 | 无 |
| GET | `/redoc` | ReDoc 文档 | 无 |
| POST | `/api/v1/chair/adjust` | 获取座椅调整建议 | 无 |
| GET | `/api/v1/chair/demo` | 演示接口 | 无 |
| POST | `/api/v1/chair/batch-adjust` | 批量调整建议 | 无 |
| GET | `/api/v1/chair/quick-adjust` | 快速调整 (简化版) | 无 |
| GET | `/api/v1/health` | 健康检查 | 无 |
| GET | `/api/v1/model` | 模型信息 | 无 |
| POST | `/api/v1/model/export` | 导出 ONNX 模型 | 无 |
| WS | `/ws/sensor` | 实时传感器数据接口 | 无 |
| GET | `/ws/stats` | WebSocket 统计 | 无 |
| GET | `/ws/sessions/{id}` | 会话详细信息 | 无 |

#### 6.2 SDK 封装示例 (Python)

```python
"""
人体工学座椅 AI SDK - Python 版本
简化 API 调用，提供高级抽象
"""

import requests
import json
import websocket
import threading
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass


@dataclass
class SensorData:
    """传感器数据封装"""
    pressure_matrix: List[List[float]]
    posture_angles: List[float]  # [head, shoulder, pelvis]
    sitting_duration: float  # seconds
    user_weight: float  # kg
    user_height: float  # meters
    fatigue_level: float = 0.0  # 0-1


@dataclass
class AdjustmentResult:
    """调整结果封装"""
    success: bool
    action: Dict[str, float]
    confidence: float
    comfort_score: float
    pressure_risk: str
    posture_analysis: Dict
    posture_detail: Optional[Dict]
    recommendations: List[str]


class ChairAIClient:
    """座椅 AI 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """检查服务健康状态"""
        resp = self.session.get(f"{self.base_url}/api/v1/health")
        return resp.json()
    
    def get_adjustment(self, sensor_data: SensorData) -> AdjustmentResult:
        """获取单次调整建议"""
        payload = {
            "pressure_matrix": sensor_data.pressure_matrix,
            "posture_angles": sensor_data.posture_angles,
            "sitting_duration": sensor_data.sitting_duration,
            "user_weight": sensor_data.user_weight,
            "user_height": sensor_data.user_height,
            "fatigue_level": sensor_data.fatigue_level
        }
        
        resp = self.session.post(
            f"{self.base_url}/api/v1/chair/adjust",
            json=payload
        )
        data = resp.json()
        
        return AdjustmentResult(
            success=data['success'],
            action=data['action'],
            confidence=data['confidence'],
            comfort_score=data['comfort_score'],
            pressure_risk=data['pressure_risk'],
            posture_analysis=data['posture_analysis'],
            posture_detail=data.get('posture_detail'),
            recommendations=data['recommendations']
        )
    
    def quick_adjust(self, sitting_minutes: float = 30.0,
                     discomfort: float = 0.5,
                     body_type: str = "average") -> Dict:
        """快速调整 (最小化输入)"""
        params = {
            "sitting_minutes": sitting_minutes,
            "discomfort_level": discomfort,
            "body_type": body_type
        }
        resp = self.session.get(
            f"{self.base_url}/api/v1/chair/quick-adjust",
            params=params
        )
        return resp.json()
    
    def batch_adjust(self, samples: List[SensorData]) -> List[AdjustmentResult]:
        """批量处理"""
        payload = {
            "samples": [
                {
                    "pressure_matrix": s.pressure_matrix,
                    "posture_angles": s.posture_angles,
                    "sitting_duration": s.sitting_duration,
                    "user_weight": s.user_weight,
                    "user_height": s.user_height,
                    "fatigue_level": s.fatigue_level
                }
                for s in samples
            ]
        }
        
        resp = self.session.post(
            f"{self.base_url}/api/v1/chair/batch-adjust",
            json=payload
        )
        data = resp.json()
        
        return [
            AdjustmentResult(**r) if r['success'] else None
            for r in data['results']
        ]
    
    def demo(self) -> Dict:
        """获取演示数据"""
        resp = self.session.get(f"{self.base_url}/api/v1/chair/demo")
        return resp.json()


class RealtimeMonitor:
    """实时监控客户端 (WebSocket)"""
    
    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.ws_url = f"{base_url}/ws/sensor"
        self.ws: Optional[websocket.WebSocketApp] = None
        self.session_id: Optional[str] = None
        self._callbacks = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def on_adjustment(self, callback: Callable):
        """注册调整建议回调"""
        self._callbacks['adjustment'] = callback
        return self
    
    def on_alert(self, callback: Callable):
        """注册告警回调"""
        self._callbacks['alert'] = callback
        return self
    
    def on_connected(self, callback: Callable):
        """注册连接成功回调"""
        self._callbacks['connected'] = callback
        return self
    
    def connect(self):
        """建立 WebSocket 连接"""
        def on_message(ws, message):
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'connected':
                self.session_id = data.get('session_id')
                cb = self._callbacks.get('connected')
                if cb:
                    cb(data['payload'])
                    
            elif msg_type == 'adjustment':
                cb = self._callbacks.get('adjustment')
                if cb:
                    cb(data['payload'])
                    
            elif msg_type == 'alert':
                cb = self._callbacks.get('alert')
                if cb:
                    cb(data['payload'])
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
            self._running = False
        
        def on_open(ws):
            print("WebSocket connected")
            self._running = True
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # 在后台线程运行
        self._thread = threading.Thread(target=self.ws.run_forever)
        self._thread.daemon = True
        self._thread.start()
        
        return self
    
    def send_sensor_data(self, sensor_data: SensorData):
        """发送传感器数据"""
        if not self.ws or not self._running:
            raise RuntimeError("WebSocket not connected")
        
        message = {
            "type": "sensor_data",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "payload": {
                "pressure_matrix": sensor_data.pressure_matrix,
                "posture_angles": sensor_data.posture_angles,
                "sitting_duration": sensor_data.sitting_duration,
                "user_weight": sensor_data.user_weight,
                "user_height": sensor_data.user_height,
                "fatigue_level": sensor_data.fatigue_level
            }
        }
        
        self.ws.send(json.dumps(message))
    
    def send_ping(self):
        """发送心跳"""
        if self.ws and self._running:
            self.ws.send(json.dumps({
                "type": "ping",
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }))
    
    def disconnect(self):
        """断开连接"""
        if self.ws:
            self.ws.close()
            self._running = False


# ======== 使用示例 ========

if __name__ == "__main__":
    # REST API 示例
    client = ChairAIClient("http://localhost:8000")
    
    # 检查服务状态
    health = client.health_check()
    print(f"服务状态: {health['status']}")
    
    # 获取调整建议
    sensor = SensorData(
        pressure_matrix=[[0.1]*8 for _ in range(8)],
        posture_angles=[15, -8, 10],
        sitting_duration=1800,
        user_weight=70,
        user_height=1.70,
        fatigue_level=0.4
    )
    
    result = client.get_adjustment(sensor)
    print(f"舒适度: {result.comfort_score}")
    print(f"姿态: {result.posture_detail['posture_name_cn'] if result.posture_detail else 'N/A'}")
    
    # WebSocket 示例
    monitor = RealtimeMonitor("ws://localhost:8000")
    
    monitor.on_connected(lambda info: print(f"已连接! 会话ID: {info['session_id']}"))
    
    monitor.on_adjustment(lambda payload: (
        print(f"\n收到调整建议:")
        print(f"  延迟: {payload['processing_latency_ms']}ms")
        print(f"  舒适度: {payload['comfort_score']}")
        if payload.get('posture_detail'):
            print(f"  姿态: {payload['posture_detail']['posture_name_cn']}")
    ))
    
    monitor.on_alert(lambda alert: (
        print(f"\n⚠️ 告警: {alert['message']} ({alert['severity']})")
    ))
    
    monitor.connect()
    
    # 模拟持续发送数据
    import time
    try:
        while True:
            monitor.send_sensor_data(sensor)
            time.sleep(1)  # 1Hz
    except KeyboardInterrupt:
        monitor.disconnect()
        print("\n已断开连接")
```

#### 6.3 错误处理最佳实践

```python
from requests.exceptions import RequestException
import time

class RobustChairClient:
    """带重试和容错的客户端"""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    def get_adjustment_with_retry(self, sensor_data: SensorData) -> Optional[AdjustmentResult]:
        """带重试机制的调整建议获取"""
        
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self.get_adjustment(sensor_data)
                
                if result.success:
                    return result
                else:
                    logger.warning(f"Request failed (attempt {attempt + 1})")
                    
            except RequestException as e:
                logger.error(f"Network error (attempt {attempt + 1}): {e}")
            
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY * (2 ** attempt))  # 指数退避
        
        # 所有重试失败，返回基于规则的默认结果
        logger.error("All retries failed, using fallback")
        return self._rule_based_fallback(sensor_data)
    
    def _rule_based_fallback(self, sensor_data: SensorData) -> AdjustmentResult:
        """规则-based 降级方案"""
        angles = sensor_data.posture_angles
        
        action = {
            "seat_height": 0.0,
            "backrest_angle": 0.0,
            "lumbar_position": 0.0,
            "lumbar_thickness": 0.0,
            "headrest_position": 0.0,
            "headrest_angle": 0.0,
            "left_armrest": 0.0,
            "right_armrest": 0.0
        }
        
        # 简单规则
        if abs(angles[0]) > 20:
            action["headrest_position"] = -0.2 * np.sign(angles[0])
        if abs(angles[1]) > 15:
            action["backrest_angle"] = -0.15 * np.sign(angles[1])
        
        return AdjustmentResult(
            success=True,
            action=action,
            confidence=0.5,  # 低置信度表示这是降级结果
            comfort_score=70.0,
            pressure_risk="unknown",
            posture_analysis={},
            posture_detail=None,
            recommendations=["服务暂时不可用，使用了基础规则"]
        )


class CircuitBreaker:
    """熔断器模式 - 防止级联故障"""
    
    states = ['CLOSED', 'OPEN', 'HALF_OPEN']
    
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = 'CLOSED'
        self.last_failure_time = None
    
    def call(self, func, *args, **kwargs):
        """通过熔断器调用函数"""
        
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            
            raise e
```

#### 6.4 集成示例: IoT 座椅固件

```c
/*
 * 智能座椅固件集成示例 (C/C++)
 * 适用于 ESP32/STM32 等微控制器
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <WebSocketsClient.h>

const char* API_BASE_URL = "http://your-server.com";
const char* WS_URL = "ws://your-server.com/ws/sensor";

// 传感器数据结构
typedef struct {
    float pressure_matrix[8][8];
    float posture_angles[3];  // head, shoulder, pelvis
    float sitting_duration;
    float user_weight;
    float user_height;
    float fatigue_level;
} SensorData_t;

// 座椅执行器
typedef struct {
    float seat_height;      // 0-100%
    float backrest_angle;   // 0-180°
    float lumbar_pos;       // 0-100mm
    float lumbar_thick;     // 0-50mm
    float headrest_pos;     // 0-200mm
    float headrest_angle;   // -30~30°
    float left_armrest;     // 0-150mm
    float right_armrest;    // 0-150mm
} ChairAction_t;

// REST API 调用
bool get_adjustment_rest(SensorData_t* sensor, ChairAction_t* action) {
    HTTPClient http;
    http.begin(API_BASE_URL "/api/v1/chair/adjust");
    http.addHeader("Content-Type", "application/json");
    
    // 构建 JSON 请求体
    StaticJsonDocument<1024> doc;
    JsonArray pressure = doc["pressure_matrix"].to<JsonArray>();
    for (int i = 0; i < 8; i++) {
        JsonArray row = pressure.createNestedArray();
        for (int j = 0; j < 8; j++) {
            row.add(sensor->pressure_matrix[i][j]);
        }
    }
    
    JsonArray angles = doc["posture_angles"].to<JsonArray>();
    for (int i = 0; i < 3; i++) {
        angles.add(sensor->posture_angles[i]);
    }
    
    doc["sitting_duration"] = sensor->sitting_duration;
    doc["user_weight"] = sensor->user_weight;
    doc["user_height"] = sensor->user_height;
    doc["fatigue_level"] = sensor->fatigue_level;
    
    String requestBody;
    serializeJson(doc, requestBody);
    
    int httpResponseCode = http.POST(requestBody);
    
    if (httpResponseCode == 200) {
        String response = http.getString();
        
        StaticJsonDocument<2048> responseDoc;
        deserializeJson(responseDoc, response);
        
        // 解析动作
        JsonObject act = responseDoc["action"];
        action->seat_height = act["seat_height"];
        action->backrest_angle = act["backrest_angle"];
        action->lumbar_pos = act["lumbar_position"];
        action->lumbar_thick = act["lumbar_thickness"];
        action->headrest_pos = act["headrest_position"];
        action->headrest_angle = act["headrest_angle"];
        action->left_armrest = act["left_armrest"];
        action->right_armrest = act["right_armrest"];
        
        http.end();
        return true;
    }
    
    http.end();
    return false;
}

// WebSocket 实时模式
WebSocketsClient webSocket;

void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            Serial.println("WS Disconnected!");
            break;
            
        case WStype_CONNECTED:
            Serial.println("WS Connected!");
            break;
            
        case WStype_TEXT: {
            Serial.printf("WS Message: %s\n", payload);
            
            StaticJsonDocument<1024> doc;
            deserializeJson(doc, (char*)payload);
            
            const char* msgType = doc["type"];
            
            if (strcmp(msgType, "adjustment") == 0) {
                JsonObject actPayload = doc["payload"]["action"];
                apply_action_from_json(actPayload);
            }
            else if (strcmp(msgType, "alert") == 0) {
                trigger_alert(doc["payload"]);
            }
            break;
        }
    }
}

void setup_websocket() {
    webSocket.begin(WS_URL);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
}

void loop_websocket(SensorData_t* sensor) {
    webSocket.loop();
    
    static unsigned long lastSend = 0;
    if (millis() - lastSend > 100) {  // 10Hz
        lastSend = millis();
        
        StaticJsonDocument<512> doc;
        doc["type"] = "sensor_data";
        doc["payload"]["pressure_matrix"] = ...;  // 填充数据
        doc["payload"]["posture_angles"] = ...;
        // ... 其他字段
        
        String output;
        serializeJson(doc, output);
        webSocket.sendTXT(output);
    }
    
    static unsigned long lastPing = 0;
    if (millis() - lastPing > 30000) {  // 心跳 30s
        lastPing = millis();
        webSocket.sendTXT("{\"type\":\"ping\"}");
    }
}

void apply_action_from_json(JsonObject& action) {
    // 映射 [-1, 1] 到实际执行器范围
    set_seat_height(map(action["seat_height"], -1, 1, 0, 100));
    set_backrest_angle(map(action["backrest_angle"], -1, 1, 90, 135));
    // ... 其他执行器
}
```

### ✅ 场景五检查清单

- [ ] 阅读 API 文档 (/docs)
- [ ] 选择合适的接口 (REST vs WebSocket)
- [ ] 实现错误处理和重试机制
- [ ] 添加熔断器和降级方案
- [ ] 处理边界情况和异常数据
- [ ] 性能测试和优化
- [ ] 日志记录和监控
- [ ] 安全性考虑 (HTTPS、认证)

---

## 7. 场景六：运维工程师 - 系统监控与管理

### 👤 用户画像
- **角色**: DevOps 工程师、SRE、系统管理员
- **目标**: 保证系统稳定运行，快速定位和解决问题
- **技能要求**: Linux、Docker、监控工具、日志分析

### 🎯 使用流程

```
部署上线 → 监控告警 → 日志分析 → 性能优化 → 故障恢复
```

### 📦 功能详解

#### 7.1 健康检查端点

```bash
# 基础健康检查
GET /api/v1/health

# 响应
{
  "status": "healthy",
  "service": "Ergonomic Chair AI Training System",
  "version": "2.0.0",
  "model_loaded": true,
  "uptime_seconds": 86400.0,
  "timestamp": "2026-04-19T18:00:00.123456",
  "components": {
    "api_server": "ok",
    "model_service": "ok",
    "classifier": "ok",
    "websocket_manager": "ok"
  },
  "metrics": {
    "requests_total": 15234,
    "requests_success_rate": 0.9992,
    "avg_response_time_ms": 12.5,
    "active_ws_connections": 15,
    "memory_usage_mb": 256.3,
    "cpu_usage_percent": 23.5
  }
}
```

#### 7.2 模型信息查询

```bash
# 获取当前加载的模型信息
GET /api/v1/model

# 响应
{
  "model_name": "chair_ai_v2.0.onnx",
  "version": "2.0.0",
  "status": "loaded",
  "loaded_at": "2026-04-19T09:00:00",
  "training_info": {
    "algorithm": "PPO (Proximal Policy Optimization)",
    "policy_type": "MlpPolicy",
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "total_timesteps": 1000000
  },
  "observation_space": 20,
  "action_space": 8,
  "algorithm": "PPO",
  "onnx_info": {
    "provider": "CPUExecutionProvider",
    "model_path": "/models/chair_ai_v2.0.onnx",
    "inference_backend": "ONNX Runtime 1.16.0"
  }
}
```

#### 7.3 WebSocket 统计监控

```bash
# 获取 WebSocket 全局统计
GET /ws/stats

# 响应
{
  "total_connections_today": 145,
  "active_connections": 15,
  "max_concurrent_connections": 28,
  "average_session_duration_seconds": 1823.5,
  "messages_sent_today": 892341,
  "messages_received_today": 892120,
  "connection_errors": 3,
  "server_uptime_hours": 24.0
}

# 获取特定会话详情
GET /ws/sessions/{session_id}

# 响应
{
  "session_id": "abc-123-def",
  "connected_at": "2026-04-19T09:15:00",
  "duration_seconds": 3600,
  "messages_received": 108000,  # 30Hz × 3600s
  "messages_sent": 108050,
  "last_activity": "2026-04-19T10:15:00",
  "status": "active",
  "client_info": {
    "ip_address": "192.168.1.100",
    "user_agent": "ChairApp/2.0 (iOS)"
  }
}
```

#### 7.4 日志配置与分析

**日志级别配置**:

```python
# api/main.py 中配置
logging.basicConfig(
    level=logging.INFO,  # DEBUG/INFO/WARNING/ERROR
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler('logs/api.log')  # 文件输出
    ]
)
```

**关键日志事件**:

```
# 模型加载
INFO - Loading model from /models/chair_ai_v2.0.onnx
INFO - Model loaded successfully

# 推理过程
DEBUG - Posture classified: forward_lean (warning)
WARNING - Posture classification error: ..., using basic analysis only

# WebSocket 事件
INFO - WebSocket session started: abc-123-def
INFO - Client disconnected: abc-123-def
ERROR - Error processing message for abc-123-def: ...

# 异常情况
ERROR - Prediction error: ..., falling back to rules
ERROR - Global exception: ...
```

#### 7.5 Prometheus 监控指标 (可选扩展)

```python
# 如果需要集成 Prometheus，可以添加以下代码
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义指标
REQUEST_COUNT = Counter(
    'chair_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'chair_api_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

ACTIVE_WS_CONNECTIONS = Gauge(
    'chair_ws_active_connections',
    'Number of active WebSocket connections'
)

MODEL_INFERENCE_COUNT = Counter(
    'chair_model_inference_total',
    'Total model inferences',
    ['model_type', 'backend']
)

# 在中间件中使用
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

# 启动 Prometheus 端点
start_http_port(9090)  # http://localhost:9090/metrics
```

#### 7.6 Docker Compose 生产部署

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  chair-api:
    image: chair-ai:2.0.0
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - CHAIR_MODEL_PATH=/models/chair_ai.onnx
      - LOG_LEVEL=INFO
      - PROMETHEUS_ENABLED=true
    volumes:
      - ./models:/models:ro
      - ./logs:/var/log/chair-ai
      - ./config:/etc/chair-ai:ro
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"

  # 可选: Redis 用于 WebSocket 会话持久化
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data

  # 可选: Nginx 反向代理
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - chair-api

volumes:
  redis-data:
```

**Nginx 配置示例**:

```nginx
# nginx/nginx.conf
upstream chair_api {
    server chair-api:8000;
}

server {
    listen 80;
    server_name chair-api.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name chair-api.example.com;
    
    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    
    # REST API
    location /api/ {
        proxy_pass http://chair_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # WebSocket
    location /ws/ {
        proxy_pass http://chair_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # WebSocket 特定超时
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
    
    # 静态文件 (Swagger UI 等)
    location /docs {
        proxy_pass http://chair_api;
    }
    
    # 速率限制
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=ws:10m rate=5r/s;
    
    location /api/ {
        limit_req zone=api burst=20 nodelay;
    }
    
    location /ws/ {
        limit_req zone=ws burst=10 nodelay;
    }
}
```

#### 7.7 常见故障排查

| 问题现象 | 可能原因 | 解决方法 |
|---------|---------|---------|
| 502 Bad Gateway | API 服务未启动 | `docker logs chair-api` 查看日志 |
| WebSocket 断连 | Nginx 超时配置 | 增加 `proxy_read_timeout` |
| 推理延迟高 (>100ms) | CPU 不足或模型过大 | 升级配置或使用量化模型 |
| 内存持续增长 | 内存泄漏 | 重启服务，检查代码 |
| 模型加载失败 | 文件路径错误或损坏 | 检查 `CHAIR_MODEL_PATH` 和文件权限 |
| 429 Too Many Requests | 触发速率限制 | 调整 `limit_req` 配置 |

### ✅ 场景六检查清单

- [ ] 健康检查端点正常响应
- [ ] 日志级别和输出配置合理
- [ ] 资源限制 (CPU/内存) 设置合适
- [ ] 自动重启策略配置
- [ ] 备份和恢复方案就绪
- [ ] 监控和告警系统接入
- [ ] SSL/TLS 证书配置
- [ ] 速率限制和防滥用措施
- [ ] 定期安全扫描和补丁更新

---

## 8. 功能矩阵速查表

### 8.1 按用户场景的功能映射

| 功能模块 | 训练 | 部署 | 终端用户 | 数据分析 | 第三方集成 | 运维监控 |
|---------|------|------|----------|----------|-----------|----------|
| **PPO 训练环境** | ✅ 核心 | - | - | - | - | - |
| **奖励函数设计** | ✅ 核心 | - | - | - | - | - |
| **动态人体模型** | ✅ 重要 | - | - | 参考 | - | - |
| **TensorBoard 监控** | ✅ 核心 | - | - | - | - | - |
| **模型评估工具** | ✅ 核心 | 参考 | - | ✅ 重要 | - | - |
| **ONNX 模型导出** | ✅ 输出 | ✅ 核心 | - | - | ✅ 重要 | - |
| **模型验证工具** | - | ✅ 核心 | - | - | ✅ 重要 | - |
| **ONNX Runtime 推理** | - | ✅ 核心 | ✅ 底层 | ✅ 底层 | ✅ 底层 | - |
| **REST API (adjust)** | - | ✅ 接口 | ✅ 主要 | ✅ 核心 | ✅ 核心 | ✅ 监控 |
| **REST API (batch)** | - | - | - | ✅ 核心 | ✅ 重要 | - |
| **REST API (quick)** | - | - | ✅ 便捷 | - | ✅ 参考 | - |
| **REST API (demo)** | - | - | ✅ 体验 | - | ✅ 测试 | - |
| **WebSocket 实时** | - | ✅ 接口 | ✅ 核心 | - | ✅ 重要 | ✅ 监控 |
| **姿态分类器** | - | ✅ 核心 | ✅ 核心 | ✅ 核心 | ✅ 核心 | - |
| **8种姿态识别** | - | ✅ 功能 | ✅ 功能 | ✅ 功能 | ✅ 功能 | - |
| **严重程度分级** | - | ✅ 功能 | ✅ 功能 | ✅ 功能 | ✅ 功能 | - |
| **健康建议系统** | - | ✅ 功能 | ✅ 功能 | ✅ 功能 | ✅ 功能 | - |
| **异常检测** | - | ✅ 功能 | ✅ 功能 | ✅ 功能 | ✅ 功能 | - |
| **健康检查端点** | - | ✅ 基础 | - | - | - | ✅ 核心 |
| **模型信息查询** | - | ✅ 基础 | - | - | - | ✅ 核心 |
| **WS 统计接口** | - | - | - | - | - | ✅ 核心 |

### 8.2 技术栈依赖关系

```
训练阶段:
  Python 3.10+
  ├── PyBullet (物理仿真)
  ├── Stable-Baselines3 (RL算法)
  ├── Gymnasium (环境接口)
  ├── NumPy (数值计算)
  └── TensorBoard (可选, 可视化)

推理/部署阶段:
  Python 3.10+
  ├── FastAPI (Web框架)
  ├── Uvicorn (ASGI服务器)
  ├── Pydantic (数据验证)
  ├── ONNX Runtime (推理引擎)
  └── websockets (WS库)

可选组件:
  ├── Redis (会话缓存)
  ├── Prometheus (监控)
  ├── Grafana (可视化)
  ├── Nginx (反向代理)
  └── Docker (容器化)
```

### 8.3 性能参考指标

| 场景 | 指标 | 期望值 | 说明 |
|------|------|--------|------|
| **REST API** | 单次推理延迟 | < 50ms | P99 |
| **REST API** | 吞吐量 | > 100 QPS | 单实例 |
| **WebSocket** | 消息处理延迟 | < 10ms | P99 |
| **WebSocket** | 最大并发连接数 | > 1000 | 取决于内存 |
| **批量处理** | 100样本处理时间 | < 200ms | CPU密集 |
| **ONNX 推理** | 单次推理 (CPU) | < 5ms | 不含预处理 |
| **ONNX 推理** | 单次推理 (GPU) | < 1ms | NVIDIA T4 |
| **ONNX 推理** | 单次推理 (边缘) | < 25ms | 树莓派4 INT8 |
| **模型大小** | ONNX 文件体积 | < 20MB | INT8量化后 < 5MB |
| **内存占用** | API 服务常驻内存 | < 500MB | 取决于并发 |

---

## 📚 附录

### A. 相关文档链接

- [API 交互式文档 (Swagger)](http://localhost:8000/docs)
- [API 文档 (ReDoc)](http://localhost:8000/redoc)
- [OpenAPI 规范 JSON](http://localhost:8000/openapi.json)
- [项目 CHANGELOG](./CHANGELOG.md)
- [版本信息](./version.py)

### B. 常见问题 FAQ

**Q1: 如何选择 REST API 还是 WebSocket?**
- **REST API**: 适用于单次查询、低频调用、简单集成
- **WebSocket**: 适用于实时监测、高频数据流、双向通信

**Q2: 没有 GPU 可以运行吗?**
- 可以。系统默认使用 CPU 推理，延迟约 5-10ms
- 如需更高性能，可启用 CUDA 加速

**Q3: 如何处理离线场景?**
- 内置 rule-based 降级方案，无需模型也可运行
- 可预加载 ONNX 模型到本地设备

**Q4: 数据隐私如何保障?**
- 支持私有化部署，数据不出内网
- 传感器数据不持久化存储（除非用户授权）

**Q5: 支持哪些平台?**
- 云端: Linux (Ubuntu/CentOS), macOS
- 边缘: 树莓派 4, Jetson Nano
- 移动: iOS (CoreML), Android (NNAPI/TFLite)

### C. 联系方式

- **项目仓库**: [GitHub链接]
- **问题反馈**: [Issues页面]
- **Email**: harleywang2000@hotmail.com
- **技术支持**: harleywang2000@hotmail.com

---

## 9. 场景七：分布式训练工程师 - 大规模模型训练 (v2.3.0 新增) ⭐

### 👤 用户画像
- **角色**: ML 工程师、分布式系统工程师、算力平台管理员
- **目标**: 利用多节点集群加速训练，管理大规模训练任务，监控系统健康状态
- **技能要求**: Python、分布式系统概念、REST API

### 🎯 使用流程

```
环境准备 → 配置协调器 → 启动Workers → 监控训练 → 模型导出
```

### 📦 功能详解

#### 9.1 本地模拟训练（快速验证）

**适用场景**: 开发调试、算法验证、无需真实集群

```python
from training.worker import LocalDistributedSimulator

# 创建4个Worker的本地模拟
simulator = LocalDistributedSimulator(
    n_workers=4,
    n_envs_per_worker=8,
    mode="sync"  # sync/async/ssp
)

# 启动并运行训练
simulator.start()
result = simulator.train(total_timesteps=50000)

# 输出结果
print(f"✅ 训练完成!")
print(f"   总样本: {result['total_samples']:,}")
print(f"   总回合: {result['total_episodes']:,}")
print(f"   吞吐量: {result['throughput_fps']:,.0f} FPS")
print(f"   用时: {result['training_time_seconds']:.1f}s")

# Worker统计
for stat in result['worker_stats']:
    print(f"   {stat['worker_id']}: {stat['samples']:,} samples, {stat['episodes']} episodes")

# 停止
simulator.stop()
```

#### 9.2 通过 REST API 管理（生产环境）

**启动训练任务**:

```bash
# 使用预设模板快速启动
curl -X POST http://localhost:8000/api/v1/training/distributed/simulate/local \
  -H "Content-Type: application/json" \
  -d '{
    "n_workers": 8,
    "n_envs_per_worker": 16,
    "total_timesteps": 1000000,
    "mode": "async"
  }'
```

**查询训练状态**:

```bash
# 全局集群状态
curl http://localhost:8000/api/v1/training/distributed/cluster/status | jq

# 特定任务详情
curl http://localhost:8000/api/v1/training/distributed/status/{job_id} | jq

# Worker健康状态
curl http://localhost:8000/api/v1/training/distributed/workers/health | jq

# 训练指标历史
curl "http://localhost:8000/api/v1/training/distributed/metrics/{job_id}?last_n=50" | jq
```

**响应示例 - 集群状态**:

```json
{
  "active_jobs": 3,
  "total_workers": 24,
  "active_workers": 23,
  "jobs": [
    {
      "job_id": "job_a1b2c3d4",
      "status": "running",
      "total_workers": 8,
      "active_workers": 8,
      "error_workers": 0,
      "timesteps": 456789,
      "episodes": 4567,
      "fps": 12500.5,
      "elapsed_seconds": 3600.2,
      "worker_details": [
        {"id": "worker_0", "state": "collecting", "steps": 60000},
        {"id": "worker_1", "state": "collecting", "steps": 58900}
      ]
    }
  ],
  "timestamp": "2026-04-19T18:30:00"
}
```

#### 9.3 高级配置：自定义分布式参数

```python
from training.distributed_trainer import (
    DistributedTrainer, 
    DistributedConfig, 
    TrainingMode, 
    AggregationMethod,
    WorkerConfig
)
from training.monitor import TrainingMonitor, AlertRule

# 创建配置
config = DistributedConfig(
    # 训练模式
    mode=TrainingMode.ASYNC,              # 异步模式 (适合异构集群)
    
    # Worker配置
    n_workers=8,                            # 8个节点
    n_envs_per_worker=32,                   # 每节点32个并行环境
    
    # 聚合策略
    aggregation_method=AggregationMethod.WEIGHTED,  # 按数据量加权
    sync_interval=200,                       # 每200步同步一次
    
    # PPO超参
    learning_rate=1e-4,                     # 学习率
    batch_size=128,                         # 批次大小
    n_epochs=15,                            # 每次更新迭代次数
    
    # 容错配置
    auto_restart_workers=True,              # 自动重启失败Worker
    heartbeat_timeout_seconds=30.0,         # 心跳超时30秒
    checkpoint_interval=5000                # 每5000步保存checkpoint
)

# 创建协调器
trainer = DistributedTrainer(config=config)
trainer.start()

# 注册Workers (通常由各节点自动连接)
for i in range(8):
    worker_config = WorkerConfig(
        worker_id=f"worker_{i}",
        n_envs=32,
        gpu_id=i % 4 if i < 8 else None     # 分配GPU (前4个)
    )
    status = trainer.register_worker(worker_config)
    print(f"注册 {status.worker_id}: {status.state.value}")

# 设置监控
monitor = TrainingMonitor()
monitor.start()

# 自定义告警规则
monitor.alert_rules.append(AlertRule(
    name="low_throughput_alert",
    condition=lambda m: m.fps < 500 and m.timestep > 10000,
    severity="warning",
    message_template="吞吐量过低警告: {fps:.1f} FPS"
))

def on_metrics(metrics):
    print(f"[{metrics.timestamp}] "
          f"Iter={metrics.iteration}, "
          f"Loss={metrics.total_loss:.4f}, "
          f"Reward={metrics.mean_episode_reward:.2f}, "
          f"FPS={metrics.fps:.0f}")

monitor.add_callback(on_metrics)
trainer.add_callback(lambda m: monitor.record_metrics(m))

# ... 等待Workers连接并开始训练 ...

# 定期检查状态
import time
while trainer.is_running:
    status = trainer.get_cluster_status()
    health = trainer.check_worker_health()
    
    healthy = sum(1 for v in health.values() if v == "healthy")
    print(f"[{status.elapsed_seconds:.0f}s] "
          f"Workers: {healthy}/{status.total_workers}, "
          f"FPS: {status.throughput_fps:.0f}, "
          f"Timesteps: {status.total_timesteps:,}")
    
    time.sleep(60)  # 每分钟检查一次

# 停止训练
final_status = trainer.stop()
monitor.stop()

print(f"\n训练完成!")
print(f"总步数: {final_status.total_timesteps:,}")
print(f"总回合: {final_status.total_episodes:,}")
print(f"用时: {final_status.elapsed_seconds:.1f}s")
```

#### 9.4 三种训练模式对比与选择

| 模式 | 一致性保证 | 吞吐量 | 适用场景 |
|------|-----------|--------|----------|
| **SYNC (同步)** | ⭐⭐⭐ 最高 | 中等 | 小规模同构集群、实验研究 |
| **ASYNC (异步)** | ⭐ 低 | ⭐⭐⭐ 最高 | 大规模异构环境、生产训练 |
| **SSP (滞留同步)** | ⭐⭐ 中等 | ⭐⭐ 高 | 平衡场景、推荐默认 |

**选择建议**:
- **开发调试/小规模 (<4 workers)**: 使用 `sync` 模式
- **生产训练/大规模 (>8 workers)**: 使用 `async` 模式
- **需要一致性但又要效率**: 使用 `ssp` 模式

#### 9.5 监控仪表板集成

```python
from training.monitor import TrainingMonitor

monitor = TrainingMonitor()
monitor.start()

# 收集指标后生成可视化数据
dashboard = monitor.generate_dashboard_data()

# Loss曲线 (前端图表库可直接使用)
loss_curve = dashboard["loss_curve"]
# [{"x": 0, "y": 0.81}, {"x": 1, "y": 0.78}, ...]

# 奖励分布
reward_dist = dashboard["reward_distribution"]
# {"mean": 7.25, "min": 5.0, "max": 9.5, "std": 1.44}

# 性能摘要
summary = dashboard["summary"]
# {"avg_loss_50episodes": 0.675, "avg_reward_50episodes": 7.25, ...}

# FPS趋势
fps_trend = dashboard["fps_trend"]

# 告警记录
alerts = dashboard["recent_alerts"]

# 导出为JSON供前端使用
import json
with open("dashboard_data.json", "w") as f:
    json.dump(dashboard, f, indent=2, default=str)
```

#### 9.6 容错机制实战

```python
# 场景1: Worker意外断开
health = trainer.check_worker_health()
stale_workers = [wid for wid, state in health.items() if state == "stale"]
if stale_workers:
    print(f"发现断开的Worker: {stale_workers}")
    if config.auto_restart_workers:
        for wid in stale_workers:
            result = trainer.restart_worker(wid)
            print(f"重启 {wid}: {'成功' if result else '失败'}")

# 场景2: 训练发散检测
recent_metrics = trainer.get_training_metrics(last_n=10)
if recent_metrics:
    latest = recent_metrics[-1]
    if latest.total_loss > 1000:
        print("⚠️ 训练可能发散! Loss过高")
        # 可以选择降低学习率或停止训练

# 场景3: Checkpoint恢复
checkpoint_path = "./checkpoints/distributed_backup.pkl"
trainer.save_checkpoint(checkpoint_path)
print(f"Checkpoint已保存到 {checkpoint_path}")

# 后续可从checkpoint恢复
new_trainer = DistributedTrainer(config=config)
if new_trainer.load_checkpoint(checkpoint_path):
    print("从Checkpoint恢复成功!")
    new_trainer.start()
```

### ✅ 场景七检查清单

- [ ] 选择合适的训练模式 (sync/async/ssp)
- [ ] 根据硬件资源配置 Worker 数量和环境数
- [ ] 选择梯度聚合策略 (mean/weighted/fedavg)
- [ ] 配置心跳超时和容错机制
- [ ] 设置监控告警规则
- [ ] 规划Checkpoint保存策略
- [ ] 测试本地模拟后再部署到集群
- [ ] 监控训练过程中的Loss/Reward/FPS指标

---

## 10. 功能矩阵速查表 (v2.3.0 更新)

### 10.1 按用户场景的功能映射

| 功能模块 | 训练 | 部署 | 终端用户 | 数据分析 | 第三方集成 | 运维监控 | 分布式训练 |
|---------|------|------|----------|----------|-----------|----------|-----------|
| **PPO 训练环境** | ✅ 核心 | - | - | - | - | - | - |
| **奖励函数设计** | ✅ 核心 | - | - | - | - | - | - |
| **动态人体模型** | ✅ 重要 | - | - | 参考 | - | - | - |
| **TensorBoard 监控** | ✅ 核心 | - | - | - | - | - | - |
| **模型评估工具** | ✅ 核心 | 参考 | - | ✅ 重要 | - | - | - |
| **ONNX 模型导出** | ✅ 输出 | ✅ 核心 | - | - | ✅ 重要 | - | - |
| **REST API** | - | ✅ 接口 | ✅ 主要 | ✅ 核心 | ✅ 核心 | ✅ 监控 | ✅ 核心 |
| **WebSocket 实时** | - | ✅ 接口 | ✅ 核心 | - | ✅ 重要 | ✅ 监控 | - |
| **姿态分类器** | - | ✅ 功能 | ✅ 功能 | ✅ 功能 | ✅ 功能 | - | - |
| **8种姿态识别** | - | ✅ 功能 | ✅ 功能 | ✅ 功能 | ✅ 功能 | - | - |
| **自定义奖励配置** | ✅ 重要 | - | - | - | - | - | - |
| **分布式协调器** | ✅ 核心 | - | - | - | - | - | ✅ 核心 |
| **Worker节点** | ✅ 核心 | - | - | - | - | - | ✅ 核心 |
| **训练监控仪表板** | ✅ 重要 | - | - | - | - | ✅ 核心 | ✅ 核心 |
| **本地模拟器** | ✅ 测试 | - | - | - | - | - | ✅ 测试 |

### 10.2 版本功能演进

| 版本 | 核心新增 | 影响角色 |
|------|---------|---------|
| **v1.0.0** | 基础PPO训练 | 算法研究员 |
| **v2.0.0** | Web API + WebSocket + ONNX | 产品工程师、开发者 |
| **v2.1.0** | 8种细粒度姿态识别 | 终端用户、数据分析 |
| **v2.2.0** | 自定义奖励函数配置 | 算法研究员 |
| **v2.3.0** | ⭐ 分布式训练支持 | ML工程师、算力管理员 |

---

> **文档版本**: v2.3.0 | **最后更新**: 2026-04-19 | **作者**: Harley Wang (王华)
