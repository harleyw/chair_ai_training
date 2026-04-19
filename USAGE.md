# 人体工学座椅 AI 训练系统 - 详细使用手册

## 目录

1. [系统概述](#系统概述)
2. [平台要求](#平台要求)
3. [环境安装](#环境安装)
4. [快速开始](#快速开始)
5. [核心概念](#核心概念)
6. [参数调整指南](#参数调整指南)
7. [训练流程详解](#训练流程详解)
8. [模型评估](#模型评估)
9. [模型部署](#模型部署)
10. [高级用法](#高级用法)
11. [常见问题](#常见问题)

***

## 系统概述

### 功能简介

本系统使用强化学习（PPO算法）训练人体工学座椅的自适应调整策略。座椅能够根据用户的：

- **坐姿**（前倾、后仰、侧倾等）
- **静坐时长**
- **体重和体型**

自动调整座垫高度、靠背角度、腰托位置等8个部件，以提供最佳的舒适度和健康支撑。

### 系统架构

```
┌─────────────────────────────────────────────┐
│                 训练系统                      │
├─────────────────────────────────────────────┤
│  环境层 (env/)                               │
│  ├─ 座椅模型 (chair_model.py)               │
│  ├─ 人体模型 (human_model/human_model.py)   │
│  ├─ 传感器系统 (sensors/sensors.py)         │
│  └─ Gymnasium环境 (chair_env/environment.py)│
├─────────────────────────────────────────────┤
│  训练层 (training/)                          │
│  ├─ PPO训练器 (train.py)                    │
│  ├─ BodyTypeRandomizer (用户随机化)          │
│  └─ 回调函数 (Checkpoint/Eval/Training)      │
├─────────────────────────────────────────────┤
│  脚本层                                      │
│  ├─ train.py (训练入口)                     │
│  ├─ evaluate.py (评估脚本)                  │
│  └─ test_*.py (测试脚本)                    │
└─────────────────────────────────────────────┘
```

***

## 平台要求

### 训练软件运行的平台要求

#### 操作系统

| 平台                        | 支持状态 | 说明                |
| ------------------------- | ---- | ----------------- |
| **Linux (Ubuntu 20.04+)** | ✅ 推荐 | 完整支持，性能最佳         |
| **macOS 10.15+**          | ✅ 支持 | 完整支持，GPU加速有限      |
| **Windows 10/11**         | ✅ 支持 | 需要WSL2或原生Python环境 |

#### Python版本

- **最低版本**: Python 3.8
- **推荐版本**: Python 3.10+
- **测试通过**: Python 3.12

#### 硬件要求

| 硬件       | 最低配置      | 推荐配置                    |
| -------- | --------- | ----------------------- |
| **CPU**  | 2核心       | 4核心+                    |
| **内存**   | 4GB       | 8GB+                    |
| **GPU**  | 无（CPU可运行） | NVIDIA GPU (CUDA 11.0+) |
| **磁盘空间** | 5GB       | 10GB+                   |

**GPU说明**:

- 训练支持GPU加速，可显著提升训练速度
- 使用 `pip install torch --index-url https://download.pytorch.org/whl/cu118` 安装CUDA版本
- 使用 `--no-gpu` 参数可强制使用CPU训练

#### 依赖库

| 依赖包               | 版本要求     | 说明       |
| ----------------- | -------- | -------- |
| pybullet          | >=3.2.5  | 物理引擎     |
| numpy             | >=1.21.0 | 数值计算     |
| torch             | >=1.10.0 | 深度学习框架   |
| gymnasium         | >=0.28.0 | 强化学习环境标准 |
| stable-baselines3 | >=2.0.0  | PPO算法实现  |
| matplotlib        | >=3.5.0  | 可视化      |

***

### 模型运行的平台要求

#### 模型格式

- **原生格式**: `.zip` (Stable-Baselines3 PPO模型)
- **ONNX格式**: `.onnx` (可选导出)

#### 推理环境要求

##### Python环境

- **最低版本**: Python 3.8
- **必需依赖**:
  - stable-baselines3 >=2.0.0
  - pybullet >=3.2.5
  - numpy >=1.21.0
  - torch >=1.10.0 (CPU版本即可)

##### 硬件要求

| 硬件       | 最低配置  | 推荐配置     |
| -------- | ----- | -------- |
| **CPU**  | 单核    | 双核+      |
| **内存**   | 1GB   | 2GB+     |
| **GPU**  | 不需要   | 可选（加速推理） |
| **磁盘空间** | 100MB | 500MB    |

##### 部署方式

| 部署方式             | 说明               |
| ---------------- | ---------------- |
| **Python脚本**     | 直接加载 `.zip` 模型文件 |
| **Web API**      | 封装为REST API服务    |
| **ONNX Runtime** | 导出为ONNX格式，跨平台部署  |
| **嵌入式设备**        | 需要适配目标平台依赖       |

##### 跨平台兼容性

- 训练好的模型可以在 **不同操作系统** 之间加载使用
- 模型文件与训练时的硬件无关，**CPU训练的模型可在GPU上推理**
- 确保推理环境的依赖版本兼容即可

##### 最小部署示例

```python
from stable_baselines3 import PPO
from env.chair_env.environment import ErgonomicChairEnv

# 加载模型（跨平台）
model = PPO.load("chair_ppo_final.zip")

# 创建环境
env = ErgonomicChairEnv()
obs, info = env.reset()

# 推理
action, _ = model.predict(obs, deterministic=True)
```

***

## 环境安装

### 方式一：使用安装脚本（推荐）

```bash
cd chair_ai_training
bash setup_cloud.sh
```

### 方式二：手动安装

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
```

### 依赖说明

| 依赖包               | 版本要求     | 用途       |
| ----------------- | -------- | -------- |
| pybullet          | >=3.2.5  | 物理引擎     |
| numpy             | >=1.21.0 | 数值计算     |
| torch             | >=1.10.0 | 深度学习框架   |
| gymnasium         | >=0.28.0 | 强化学习环境标准 |
| stable-baselines3 | >=2.0.0  | PPO算法实现  |
| matplotlib        | >=3.5.0  | 可视化      |

### 验证安装

```bash
source venv/bin/activate
python3 -c "import pybullet; import stable_baselines3; import torch; print('安装成功')"
```

***

## 快速开始

### 1. 测试环境

```bash
python test_env.py
```

预期输出：

```
Creating environment...
Testing reset...
  Observation shape: (20,)
Testing random actions...
  Step 1: Reward = 0.1234, Posture = neutral
...
All environment tests passed!
```

### 2. 测试训练流程

```bash
python test_training.py
```

### 3. 开始训练

```bash
# 快速测试（1000步）
python train.py --timesteps 1000

# 标准训练（10万步）
python train.py --timesteps 100000
```

### 4. 评估模型

```bash
python evaluate.py --model-path ./models/chair_ppo_final_*.zip
```

***

## 核心概念

### 状态空间（Observation Space）- 20维

| 维度    | 内容    | 说明                          |
| ----- | ----- | --------------------------- |
| 0-7   | 座椅状态  | 座高、靠背角、腰托位置/厚度、头枕位置/角、左右扶手高 |
| 8-10  | 姿态角度  | 俯仰(pitch)、翻滚(roll)、偏航(yaw)  |
| 11-13 | 传感器信息 | 头部、肩部、骨盆位置                  |
| 14    | 静坐时长  | 当前姿态持续时间（秒）                 |
| 15    | 用户体重  | kg                          |
| 16    | 体型类型  | 0.0(瘦) - 1.0(胖)             |
| 17    | 疲劳度   | 0.0(无疲劳) - 1.0(最大疲劳)        |
| 18    | 平均压力  | 压力传感器平均值                    |
| 19    | 最大压力  | 压力传感器最大值                    |

### 动作空间（Action Space）- 8维

| 动作索引 | 控制部件  | 调整范围（每步） |
| ---- | ----- | -------- |
| 0    | 座垫高度  | ±5mm     |
| 1    | 靠背角度  | ±0.57°   |
| 2    | 腰托位置  | ±5mm     |
| 3    | 腰托厚度  | ±2mm     |
| 4    | 头枕高度  | ±5mm     |
| 5    | 头枕角度  | ±0.57°   |
| 6    | 左扶手高度 | ±5mm     |
| 7    | 右扶手高度 | ±5mm     |

动作值范围：`[-1.0, 1.0]`，实际调整量 = 动作值 × 最大调整幅度

### 奖励函数

```
R_total = 1.0 × R_舒适 - 0.8 × R_压力 - 0.5 × R_静态 - 0.3 × R_能耗
```

| 组成部分      | 权重  | 计算方式                   |
| --------- | --- | ---------------------- |
| **R\_舒适** | 1.0 | 0.5×脊柱对齐 + 0.5×压力均匀分布  |
| **R\_压力** | 0.8 | max(0, (最大压力-50)/50)   |
| **R\_静态** | 0.5 | max(0, (静坐时长-900)/900) |
| **R\_能耗** | 0.3 | 动作幅度 × 0.01            |

***

## 参数调整指南

### 训练超参数

```bash
python train.py \
    --timesteps 100000 \      # 总训练步数，越多训练越充分
    --n-envs 4 \              # 并行环境数，CPU建议2-4，GPU建议4-8
    --lr 3e-4 \               # 学习率，太大可能不收敛，太小训练慢
    --n-steps 2048 \          # 每次更新步数，建议256-4096
    --batch-size 64 \         # 批次大小，建议32-256
    --n-epochs 10 \           # 每次更新迭代次数，建议5-20
    --gamma 0.99 \            # 折扣因子，0-1之间
    --gae-lambda 0.95 \       # GAE参数，0-1之间
    --ent-coef 0.01 \         # 熵系数，鼓励探索
    --save-freq 10000         # 保存检查点频率
```

### 推荐配置

| 场景        | timesteps | n-envs | GPU |
| --------- | --------- | ------ | --- |
| **快速测试**  | 1,000     | 2      | 否   |
| **验证流程**  | 10,000    | 2      | 否   |
| **标准训练**  | 100,000   | 4      | 可选  |
| **生产训练**  | 500,000   | 8      | 是   |
| **长时间训练** | 1,000,000 | 8      | 是   |

### 用户特征随机化

在 `training/train.py` 中修改 `BodyTypeRandomizer` 的参数：

```python
def create_env(env_idx=0):
    from env.chair_env.environment import ErgonomicChairEnv
    base_env = ErgonomicChairEnv(render_mode=None)
    return BodyTypeRandomizer(
        base_env,
        height_range=(1.55, 1.85),    # 身高范围（米）
        weight_range=(50, 100),       # 体重范围（kg）
        body_type_range=(0.1, 0.9)    # 体型范围（0=瘦, 1=胖）
    )
```

### 环境参数

在 `env/chair_env/environment.py` 中调整：

```python
class ErgonomicChairEnv(gym.Env):
    def __init__(self, render_mode=None):
        # ...
        self.max_steps = 1000         # 每回合最大步数
        self.time_step = 1.0          # 模拟时间步长（秒）
        
        # 奖励权重
        self.w_comfort = 1.0          # 舒适奖励权重
        self.w_pressure = 0.8         # 压力惩罚权重
        self.w_static = 0.5           # 久坐惩罚权重
        self.w_energy = 0.3           # 能耗惩罚权重
```

### 人体特征参数

在 `env/human_model/human_model.py` 中调整：

```python
class HumanModel:
    def __init__(self, ...):
        self.fatigue_rate = 0.0005         # 疲劳累积速率
        self.max_fatigue = 1.0             # 最大疲劳度
        self.com_shift_frequency = 0.01    # 重心偏移频率
        self.com_shift_magnitude = 0.02    # 重心偏移幅度
        self.micro_movement_prob = 0.05    # 微动概率
```

***

## 训练流程详解

### 训练循环示意图

```
┌──────────────┐
│  env.reset() │ ← 随机生成新用户（身高、体重、体型）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 获取初始状态  │ ← 20维观察向量
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ PPO选择动作A    │ ← 8维动作向量
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 环境执行动作     │ ← 座椅调整部件
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 计算奖励R       │ ← 舒适度-压力-静态-能耗
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 更新疲劳/重心   │ ← 动态人体特征
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ terminated或     │ ← 达到最大步数？
│ truncated？      │
└──────┬───────────┘
       │
    是 ↓  否
┌──────────┐    ┌──────────────┐
│ env.reset│    │ 继续下一步骤  │
│ (新回合) │    └──────────────┘
└──────────┘
```

### 训练命令详解

#### 1. 从零开始训练

```bash
python train.py --timesteps 100000 --n-envs 4
```

#### 2. 使用GPU训练

```bash
# 自动检测GPU（默认开启）
python train.py --timesteps 100000

# 强制使用CPU
python train.py --timesteps 100000 --no-gpu
```

#### 3. 断点续训

```bash
python train.py \
    --load-path ./models/chair_ppo_20260417_123456_50000_steps.zip \
    --timesteps 50000
```

#### 4. 自定义日志和模型目录

```bash
python train.py \
    --timesteps 100000 \
    --log-dir /data/logs \
    --model-dir /data/models
```

### 训练输出

```
2026-04-17 12:00:00 [INFO] ============================================
2026-04-17 12:00:00 [INFO] Ergonomic Chair AI Training
2026-04-17 12:00:00 [INFO] ============================================
2026-04-17 12:00:00 [INFO] Total Timesteps: 100,000
2026-04-17 12:00:00 [INFO] Using device: cpu
2026-04-17 12:00:00 [INFO] Starting training...
2026-04-17 12:00:10 [INFO] Episode 10: Reward=5.23, Mean(100)=4.85, Length=1000
2026-04-17 12:01:00 [INFO] Timesteps: 10,000, FPS: 165, Elapsed: 1.0 min
...
2026-04-17 12:10:00 [INFO] Training completed in 10.5 minutes
2026-04-17 12:10:00 [INFO] Final model saved to ./models/chair_ppo_final_20260417_120000
```

### 训练产物

训练完成后会生成：

```
logs/
└── run_20260417_120000/
    ├── events.out.tfevents.*     # TensorBoard日志
    ├── eval/                     # 评估日志
    ├── training_progress_*.png   # 训练进度图
    └── training_summary.json     # 训练总结

models/
├── best_model/                   # 最佳模型（自动评估选出）
│   └── best_model.zip
├── chair_ppo_20260417_120000_10000_steps.zip  # 检查点
├── chair_ppo_20260417_120000_20000_steps.zip
└── chair_ppo_final_20260417_120000.zip        # 最终模型
```

***

## 模型评估

### 基本评估

```bash
python evaluate.py \
    --model-path ./models/chair_ppo_final_*.zip \
    --n-episodes 10
```

### 可视化评估

```bash
python evaluate.py \
    --model-path ./models/chair_ppo_final_*.zip \
    --n-episodes 10 \
    --render
```

### 评估输出

```
2026-04-17 12:20:00 [INFO] Loading model from ./models/chair_ppo_final_*.zip
2026-04-17 12:20:01 [INFO] Episode 1/10: Reward = 45.67, Length = 1000
2026-04-17 12:20:02 [INFO] Episode 2/10: Reward = 48.23, Length = 1000
...
2026-04-17 12:20:10 [INFO] ============================================
2026-04-17 12:20:10 [INFO] Evaluation Results:
2026-04-17 12:20:10 [INFO] Mean Reward: 46.85 ± 2.34
2026-04-17 12:20:10 [INFO] Mean Length: 1000.00 ± 0.00
2026-04-17 12:20:10 [INFO] Max Reward: 51.23
2026-04-17 12:20:10 [INFO] Min Reward: 42.10
2026-04-17 12:20:10 [INFO] ============================================
```

***

## 模型部署

### 1. 在Python中使用训练好的模型

```python
from stable_baselines3 import PPO
from env.chair_env.environment import ErgonomicChairEnv

# 加载模型
model = PPO.load("./models/chair_ppo_final_20260417_120000.zip")

# 创建环境
env = ErgonomicChairEnv(render_mode=None)
obs, info = env.reset()

# 运行策略
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

### 2. 仅使用策略推理

```python
# 获取当前座椅状态和用户信息
current_state = get_sensor_data()  # 20维向量

# 预测最佳动作
action, _ = model.predict(current_state, deterministic=True)

# 应用动作到真实座椅
apply_to_real_chair(action)
```

### 3. 导出为ONNX（可选）

```python
import torch

# 加载模型
model = PPO.load("./models/chair_ppo_final.zip")

# 导出策略网络
dummy_input = torch.randn(1, 20)
torch.onnx.export(
    model.policy,
    dummy_input,
    "chair_policy.onnx",
    input_names=["observation"],
    output_names=["action"]
)
```

***

## 高级用法

### 1. TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir ./logs --port 6006

# 浏览器访问 http://localhost:6006
```

### 2. 自定义奖励权重

```python
# 在训练前调整权重
env = ErgonomicChairEnv()
env.w_comfort = 1.5   # 更强调舒适度
env.w_energy = 0.1    # 降低能耗惩罚
```

### 3. 多用户场景训练

```python
# 自定义环境创建函数
def create_custom_env():
    env = ErgonomicChairEnv()
    # 可以添加更多自定义逻辑
    return env
```

### 4. 训练监控脚本

```python
# 监控训练进度
import json

with open('./logs/run_*/training_summary.json') as f:
    summary = json.load(f)
    print(f"Mean Reward: {summary['mean_reward']}")
    print(f"Training Time: {summary['training_time_minutes']} min")
```

***

## 常见问题

### Q1: 训练时出现 `No module named 'pybullet'` 错误

```bash
source venv/bin/activate
pip install pybullet
```

### Q2: 训练速度太慢

- 减少 `n-steps` 和 `batch-size`
- 增加 `n-envs`（如果有足够CPU核心）
- 使用 `--no-gpu` 强制CPU（如果GPU显存不足）

### Q3: 奖励不收敛

- 降低学习率 `--lr 1e-4`
- 增加训练步数 `--timesteps 500000`
- 调整奖励权重（减少惩罚项权重）

### Q4: 如何继续训练

```bash
python train.py --load-path ./models/chair_ppo_final_*.zip --timesteps 100000
```

### Q5: 如何修改座椅部件范围

编辑 `env/chair_model.py` 中的参数：

```python
self.min_seat_height = 0.40  # 最小座高（米）
self.max_seat_height = 0.55  # 最大座高（米）
```

### Q6: 磁盘空间不足

```bash
# 清理旧的检查点
rm ./models/chair_ppo_*_steps.zip

# 只保留最终模型和最佳模型
ls -lh ./models/
```

### Q7: 如何测试单个组件

```bash
# 测试环境
python test_env.py

# 测试训练流程
python test_training.py

# 测试动态人体特征
python test_dynamic_human.py
```

### Q8: 训练日志在哪里

```bash
# 查看所有训练记录
ls -la ./logs/

# 查看某次训练的总结
cat ./logs/run_*/training_summary.json

# 查看TensorBoard
tensorboard --logdir ./logs
```

---

## 分布式训练使用指南 (v2.3.0 新增)

### 概述

v2.3.0 版本引入了完整的分布式训练支持，允许多个 Worker 节点协同训练模型，显著提升训练效率。

**适用场景**:
- 大规模训练任务（>100万步）
- 多 GPU / 多机器环境
- 需要快速实验和超参数搜索

### 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                  分布式训练系统架构                           │
├─────────────────────────────────────────────────────────────┤
│   ┌─────────────┐    ┌──────────┐    ┌──────────────────┐  │
│   │ Coordinator │───▶│  Monitor  │◀──│    API Server     │  │
│   │ (主节点)     │    │ (监控)    │    │  (REST 端点)      │  │
│   └──────┬──────┘    └──────────┘    └──────────────────┘  │
│          │                                                  │
│          ▼                                                  │
│   ┌────────────────────────────────────────────────────┐   │
│   │              Worker Pool (工作节点池)                │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │   │
│   │  │ Worker  │  │ Worker  │  │ Worker  │  ...         │   │
│   │  │ Node 0  │  │ Node 1  │  │ Node 2  │             │   │
│   │  │ Env×N   │  │ Env×N   │  │ Env×N   │             │   │
│   │  └────┬────┘  └────┬────┘  └────┬────┘             │   │
│   └───────┼────────────┼────────────┼─────────────────┘   │
│           ▼            ▼            ▼                      │
│   ┌────────────────────────────────────────────────────┐   │
│   │           Experience Buffer (共享经验缓冲区)         │   │
│   └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 快速开始：本地模拟

无需真实集群，在单机上模拟分布式训练：

```python
from training.worker import LocalDistributedSimulator

# 创建模拟器（4个Worker，每个8个环境）
simulator = LocalDistributedSimulator(
    n_workers=4,
    n_envs_per_worker=8,
    mode="sync"  # 可选: sync, async, ssp
)

# 启动训练
simulator.start()
result = simulator.train(total_timesteps=50000)

# 查看结果
print(f"✅ 训练完成!")
print(f"   总样本数: {result['total_samples']}")
print(f"   总回合数: {result['total_episodes']}")
print(f"   吞吐量: {result['throughput_fps']:.1f} FPS")
print(f"   训练时间: {result['training_time_seconds']:.1f}s")

# 停止并获取最终状态
final_status = simulator.stop()
```

### 通过 REST API 使用

#### 1. 启动 API 服务

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. 获取预设配置模板

```bash
curl http://localhost:8000/api/v1/training/distributed/templates | jq
```

响应示例：
```json
[
  {
    "name": "quick_test",
    "display_name": "快速测试",
    "config": {
      "n_workers": 2,
      "n_envs_per_worker": 4,
      "total_timesteps": 10000,
      "mode": "sync"
    }
  },
  {
    "name": "standard",
    "display_name": "标准训练",
    "config": {
      "n_workers": 4,
      "n_envs_per_worker": 16,
      "total_timesteps": 500000,
      "mode": "sync"
    }
  }
]
```

#### 3. 启动本地模拟训练

```bash
curl -X POST http://localhost:8000/api/v1/training/distributed/simulate/local \
  -H "Content-Type: application/json" \
  -d '{
    "n_workers": 4,
    "n_envs_per_worker": 8,
    "total_timesteps": 50000,
    "mode": "sync"
  }' | jq
```

响应示例：
```json
{
  "success": true,
  "training_time_seconds": 45.3,
  "training_time_minutes": 0.755,
  "total_samples": 409600,
  "total_episodes": 409,
  "throughput_fps": 9042.8,
  "n_workers": 4,
  "n_envs_total": 32,
  "worker_stats": [
    {"worker_id": "worker_0", "samples": 102400, "episodes": 102},
    {"worker_id": "worker_1", "samples": 102400, "episodes": 103},
    {"worker_id": "worker_2", "samples": 102400, "episodes": 102},
    {"worker_id": "worker_3", "samples": 102400, "episodes": 102}
  ]
}
```

#### 4. 查询集群状态

```bash
# 全局状态
curl http://localhost:8000/api/v1/training/distributed/cluster/status | jq

# 特定任务状态
curl http://localhost:8000/api/v1/training/distributed/status/{job_id} | jq
```

### 训练模式详解

| 模式 | 名称 | 适用场景 | 优点 | 缺点 |
|------|------|----------|------|------|
| **sync** | 同步模式 | 小规模同构集群 | 一致性最高，实现简单 | 受最慢Worker限制 |
| **async** | 异步模式 | 异构环境/大规模 | 吞吐量最大 | 可能不一致 |
| **ssp** | 滞留同步并行 | 平衡场景 | 兼顾效率和一致性 | 配置复杂 |

### 梯度聚合策略

| 策略 | 名称 | 适用场景 |
|------|------|----------|
| **mean** | 平均聚合 | 同构Worker（默认）|
| **weighted** | 加权聚合 | 异构环境（按数据量加权）|
| **fedavg** | 联邦平均 | 隐私敏感场景 |

### 高级配置示例

```python
from training.distributed_trainer import DistributedTrainer, DistributedConfig
from training import TrainingMode, AggregationMethod

# 创建自定义配置
config = DistributedConfig(
    mode=TrainingMode.ASYNC,              # 异步模式
    n_workers=8,                            # 8个Worker
    n_envs_per_worker=32,                   # 每个32个环境
    
    aggregation_method=AggregationMethod.WEIGHTED,  # 加权聚合
    sync_interval=200,                       # 每200步同步一次
    
    learning_rate=1e-4,                     # 较小学习率
    batch_size=128,                         # 较大batch
    n_epochs=15,                            # 更多epoch
    
    auto_restart_workers=True,              # 自动重启失败Worker
    checkpoint_interval=5000                # 每5000步保存checkpoint
)

# 创建协调器
trainer = DistributedTrainer(config=config)
trainer.start()

# 注册Workers
for i in range(8):
    trainer.register_worker(WorkerConfig(
        worker_id=f"worker_{i}",
        n_envs=32,
        gpu_id=i % 4  # 分配GPU
    ))

# ... Workers连接并开始训练 ...

# 获取集群状态
status = trainer.get_cluster_status()
print(f"活跃Workers: {status.active_workers}/{status.total_workers}")

# 停止训练
trainer.stop()
```

### 监控与告警

```python
from training.monitor import TrainingMonitor, AlertRule

# 创建监控实例
monitor = TrainingMonitor()
monitor.start()

# 自定义告警规则
custom_alert = AlertRule(
    name="low_reward_warning",
    condition=lambda m: m.mean_episode_reward < 0 and m.episode > 50,
    severity="warning",
    message_template="平均奖励过低: {mean_episode_reward:.2f}"
)
monitor.alert_rules.append(custom_alert)

# 添加回调（例如推送到WebSocket）
def on_metrics_update(metrics):
    print(f"[{metrics.timestamp}] Loss={metrics.total_loss:.4f}, Reward={metrics.mean_episode_reward:.2f}")

monitor.add_callback(on_metrics_update)

# 生成仪表板数据
dashboard = monitor.generate_dashboard_data()
print(dashboard["loss_curve"])       # Loss曲线数据
print(dashboard["reward_distribution"])  # 奖励分布统计
print(dashboard["summary"])          # 性能摘要
```

### 容错机制

系统内置以下容错功能：

1. **心跳检测**: 默认30秒超时
2. **自动重启**: 可配置是否自动重启失败的Worker
3. **Checkpoint持久化**: 定期保存训练状态
4. **健康检查API**: `GET /api/v1/training/distributed/workers/health`

### 性能调优建议

| 场景 | 推荐配置 | 预期吞吐量 |
|------|----------|-----------|
| 开发调试 | 2 workers × 4 envs | ~2000 FPS |
| 标准训练 | 4 workers × 16 envs | ~8000 FPS |
| 大规模训练 | 8 workers × 32 envs | ~25000 FPS |
| 超参数搜索 | 16 workers × 8 envs | ~15000 FPS |

### 测试分布式训练

```bash
# 运行完整测试套件
python test_distributed_training.py
```

预期输出：
```
🎯 分布式训练系统测试套件 v2.3.0
==========================================
✅ 经验缓冲区测试:       4/4   通过 (100%)
✅ 协调器核心功能:       7/7   通过 (100%)
✅ 梯度聚合算法:         4/4   通过 (100%)
✅ 容错机制测试:         5/5   通过 (100%)
✅ 监控服务测试:         6/6   通过 (100%)
✅ Worker 节点测试:       2/2   通过 (100%)
✅ API 配置验证:         3/3   通过 (100%)
✅ 端到端集成测试:       1/1   通过 (100%)

总计: 31/31 测试用例通过 🎉
```

---

## 附录：版本更新记录

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v2.3.0 | 2026-04-19 | ⭐ 分布式训练支持、监控仪表板 |
| v2.2.0 | 2026-04-19 | 自定义奖励函数配置系统 |
| v2.1.0 | 2026-04-19 | 8种细粒度姿态识别 |
| v2.0.0 | 2026-04-19 | Web API、WebSocket、ONNX导出 |
| v1.0.1 | 2026-04-18 | 修复兼容性问题 |
| v1.0.0 | 2026-04-17 | 初始版本发布 |

---

**文档版本**: v2.3.0 | **最后更新**: 2026-04-19 | **作者**: Harley Wang (王华) | **Email**: harleywang2000@hotmail.com

