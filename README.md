# Ergonomic Chair AI Training System

基于强化学习（PPO）的人体工学座椅自适应训练系统，使用具身智能方法训练座椅根据用户坐姿、静坐时长和体重自动调整。

**版本**: v2.3.0 | **作者**: Harley Wang (王华) | **许可证**: MIT License

## ✨ 核心特性 (v2.3.0)

### 🚀 分布式训练支持 ⭐⭐⭐
- **多节点并行训练**: 支持 1-64 个 Worker 节点协同训练
- **三种训练模式**: SYNC (同步) / ASYNC (异步) / SSP (滞留同步并行)
- **灵活的梯度聚合**: MEAN (平均) / WEIGHTED (加权) / FEDAVG (联邦平均)
- **完善的容错机制**: 心跳检测、自动重启、Checkpoint 持久化
- **实时监控仪表板**: Loss/Reward/FPS 曲线、智能告警规则
- **单机模拟开发**: 无需集群即可测试分布式逻辑
- **REST API 管理**: 14+ 个端点覆盖全生命周期

### 🎯 姿态识别系统
- **8 种坐姿类型识别**: 正常/前倾/后仰/侧偏/交叉腿/跷二郎腿/盘腿/前伸
- **4 级严重程度分级**: ideal/good/warning/danger
- **多维度特征融合**: 角度特征 + 压力矩阵派生特征

### 🎨 自定义奖励函数配置
- **可视化参数配置**: 无需修改代码即可调整所有超参数
- **5 种内置预设**: balanced/health_first/comfort_priority/strict_posture/energy_saving
- **实时预览能力**: 单点计算、曲线生成、多配置对比
- **安全自定义公式**: 沙箱执行环境 + 白名单机制
- **热更新支持**: 训练过程中动态调整策略

### 🌐 Web API 服务
- **RESTful API**: 基于 FastAPI，15+ 端点
- **WebSocket 实时通信**: 双向低延迟数据流 (<1ms, >800fps)
- **ONNX 导出支持**: 跨平台模型部署 (树莓派等边缘设备)
- **交互式文档**: Swagger UI (`/docs`) 和 ReDoc (`/redoc`)

## 📋 环境要求

### 训练平台要求
| 组件 | 版本要求 |
|------|----------|
| Python | >= 3.10 (推荐 3.12+) |
| PyBullet | >= 3.2.5 |
| PyTorch | >= 2.0.0 |
| Gymnasium | >= 0.28.0 |
| Stable-Baselines3 | >= 2.0.0 |
| NumPy | >= 1.24.0 |

### API 服务额外依赖
| 组件 | 用途 |
|------|------|
| FastAPI | Web 框架 |
| Uvicorn | ASGI 服务器 |
| Pydantic | 数据验证 |
| ONNX Runtime | 模型推理 |
| Websockets | WebSocket 支持 |

### 推理平台要求 (ONNX)
| 平台 | CPU | GPU | 内存 |
|------|-----|-----|------|
| Linux x86_64 | ✅ | 可选 | >= 2GB |
| macOS ARM64 | ✅ | MPS | >= 4GB |
| Windows x86_64 | ✅ | CUDA | >= 4GB |
| Raspberry Pi 4 | ✅ | ❌ | >= 2GB |

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 单机训练

```bash
# 基础训练
python train.py --timesteps 100000 --n-envs 4

# 使用自定义奖励函数预设
python train.py --reward-preset health_first --timesteps 500000
```

### 3. 分布式训练 (v2.3.0 新增)

```python
from training.worker import LocalDistributedSimulator

# 方式A: Python 代码调用
simulator = LocalDistributedSimulator(
    n_workers=4,
    n_envs_per_worker=8,
    mode="sync"
)

simulator.start()
result = simulator.train(total_timesteps=50000)
print(f"吞吐量: {result['throughput_fps']:.1f} FPS")
simulator.stop()

# 方式B: REST API 调用
# POST http://localhost:8000/api/v1/training/distributed/simulate/local
# Body: {"n_workers": 4, "mode": "sync", "total_timesteps": 50000}
```

### 4. 启动 API 服务

```bash
# 启动 FastAPI 服务
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 访问文档
# Swagger UI: http://localhost:8000/docs
# ReDoc:      http://localhost:8000/redoc
```

### 5. 评估与导出

```bash
# 评估模型
python evaluate.py --model-path ./models/chair_ppo_final.zip

# 导出为 ONNX 格式 (用于边缘部署)
python export_onnx.py --model-path ./models/chair_ppo_final.zip --output ./exported/
```

## 📖 项目结构

```
chair_ai_training/
├── api/                              # Web API 模块
│   ├── main.py                       # FastAPI 应用入口
│   ├── models.py                     # Pydantic 数据模型
│   ├── service.py                    # 核心业务逻辑
│   ├── posture_classifier.py         # 姿态分类器 (8种坐姿)
│   ├── reward_config.py              # 奖励函数配置系统
│   ├── ws_manager.py                 # WebSocket 连接管理
│   └── routes/                       # API 路由
│       ├── chair.py                  # 座椅控制接口
│       ├── model.py                  # 模型管理接口
│       ├── websocket.py              # WebSocket 实时数据
│       ├── reward_config.py          # 奖励配置接口 (20+端点)
│       └── distributed.py            # 分布式训练接口 (14+端点) ⭐
├── env/                              # 模拟环境
│   ├── chair_model.py                # 座椅物理模型
│   ├── chair_env/environment.py      # Gymnasium 环境
│   ├── human_model/human_model.py    # 人体生物力学模型
│   └── sensors/sensors.py            # 传感器系统
├── training/                         # 训练模块
│   ├── train.py                      # 单机训练核心
│   ├── distributed_trainer.py        # 分布式训练协调器 ⭐
│   ├── worker.py                     # Worker 节点实现 ⭐
│   ├── monitor.py                    # 训练监控仪表板 ⭐
│   └── dynamic_rewards.py            # 动态奖励函数构建器
├── export/                           # 导出工具
│   ├── exporter.py                   # ONNX 导出逻辑
│   ├── validator.py                  # 模型验证
│   └── runtime_inference.py          # ONNX Runtime 推理
├── train.py                          # 训练入口脚本
├── evaluate.py                       # 模型评估脚本
├── export_onnx.py                    # ONNX 导出 CLI
├── version.py                        # 版本定义
├── requirements.txt                  # 依赖列表
├── CHANGELOG.md                      # 版本变更记录
└── README.md                         # 本文件
```

## 🎮 训练参数说明

### 单机训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--timesteps` | 100000 | 总训练步数 |
| `--n-envs` | 4 | 并行环境数量 |
| `--lr` | 3e-4 | 学习率 |
| `--n-steps` | 2048 | PPO 每次更新的步数 |
| `--batch-size` | 64 | 批次大小 |
| `--n-epochs` | 10 | 每次更新的迭代次数 |
| `--gamma` | 0.99 | 折扣因子 |
| `--reward-preset` | balanced | 奖励函数预设 |
| `--log-dir` | ./logs | TensorBoard 日志目录 |
| `--model-dir` | ./models | 模型保存目录 |

### 分布式训练参数 (v2.3.0)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mode` | sync | 训练模式 (sync/async/ssp) |
| `n_workers` | 4 | Worker 节点数量 (1-64) |
| `n_envs_per_worker` | 16 | 每个 Worker 的环境数 |
| `aggregation_method` | mean | 梯度聚合方式 (mean/weighted/fedavg) |
| `sync_interval` | 100 | 同步间隔 (步数) |
| `learning_rate` | 3e-4 | 学习率 |
| `auto_restart` | True | Worker 失败时自动重启 |

## 📐 奖励函数设计

```
R_total = w₁ × R_舒适 - w₂ × R_压力 - w₃ × R_静态 - w₄ × R_能耗
```

### 分量说明

| 分量 | 权重范围 | 计算依据 | 优化目标 |
|------|----------|----------|----------|
| **舒适奖励** | 0.5 - 2.0 | 脊柱对齐 + 压力均匀分布 | 保持正确坐姿 |
| **压力惩罚** | 0.5 - 2.0 | 局部压强过高区域 | 避免局部压迫 |
| **静态惩罚** | 0.3 - 1.5 | 静坐时长超过阈值 | 鼓励适度活动 |
| **能耗惩罚** | 0.1 - 1.0 | 座椅调整频率和幅度 | 减少不必要的调整 |

### 内置预设

| 预设名称 | 适用场景 | comfort | pressure | static | energy |
|---------|---------|---------|----------|--------|--------|
| `balanced` | 通用办公 | 1.0 | 0.8 | 0.6 | 0.3 |
| `health_first` | 医疗康复 | 0.8 | 1.5 | 1.0 | 0.2 |
| `comfort_priority` | 长时间工作 | 1.8 | 0.6 | 0.4 | 0.5 |
| `strict_posture` | 严格工效学 | 0.9 | 1.8 | 1.2 | 0.3 |
| `energy_saving` | 低功耗设备 | 0.7 | 0.9 | 0.8 | 1.5 |

## 🧪 测试套件

```bash
# 运行所有测试
python test_env.py                           # 环境功能测试
python test_training.py                      # 训练流程测试
python test_dynamic_human.py                 # 动态人体特征测试
python test_posture_classification.py         # 姿态分类测试 (v2.1)
python test_reward_config.py                 # 奖励配置测试 (v2.2)
python test_distributed_training.py           # 分布式训练测试 (v2.3) ⭐
python test_websocket.py                     # WebSocket 测试
python test_onnx_export.py                   # ONNX 导出测试
```

### v2.3.0 测试结果

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

## 📊 API 端点概览

### 核心接口

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | API 基本信息 |
| GET | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/chair/adjust` | 获取座椅调整建议 |
| WS | `/ws/sensor` | 实时传感器数据流 ⭐ |

### 分布式训练接口 (v2.3.0 新增)

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/training/distributed/start` | 启动分布式训练 |
| GET | `/api/v1/training/distributed/status/{job_id}` | 查询任务状态 |
| POST | `/api/v1/training/distributed/pause/{job_id}` | 暂停训练 |
| POST | `/api/v1/training/distributed/stop/{job_id}` | 停止训练 |
| GET | `/api/v1/training/distributed/workers` | 列出所有 Worker |
| POST | `/api/v1/training/distributed/simulate/local` | 本地模拟训练 |
| GET | `/api/v1/training/distributed/templates` | 预设配置模板 |

详细文档请访问: `http://localhost:8000/docs`

## 📈 版本历史

查看 [CHANGELOG.md](./CHANGELOG.md) 了解详细的版本变更记录。

### 主要版本里程碑

| 版本 | 日期 | 重要特性 |
|------|------|----------|
| **v2.3.0** | 2026-04-19 | ⭐ 分布式训练支持、监控仪表板 |
| **v2.2.0** | 2026-04-19 | 自定义奖励函数配置系统 |
| **v2.1.0** | 2026-04-19 | 8 种细粒度姿态识别 |
| **v2.0.0** | 2026-04-19 | Web API、WebSocket、ONNX 导出 |
| **v1.0.1** | 2026-04-18 | 修复兼容性问题 |
| **v1.0.0** | 2026-04-17 | 初始版本发布 |

## 📝 许可证

本项目基于 [MIT License](https://opensource.org/licenses/MIT) 开源。

```
MIT License

Copyright (c) 2026 Harley Wang (王华)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👤 作者

**Harley Wang (王华)**
**Email**: harleywang2000@hotmail.com

## 🔗 相关链接

- 📖 [使用手册](./USAGE.md) - 详细的使用指南
- 🎯 [用户场景指南](./USER_SCENARIOS_GUIDE.md) - 不同角色的使用场景
- 📋 [CHANGELOG](./CHANGELOG.md) - 版本变更历史
