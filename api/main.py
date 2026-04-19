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
Ergonomic Chair AI Training - Web API Service
基于 FastAPI 的人体工学座椅智能控制系统 RESTful API

运行方式:
    python -m api.main
    或
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    
API 文档:
    Swagger UI: http://localhost:8000/docs
    ReDoc: http://localhost:8000/redoc
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import datetime

from api.service import ChairAIService
from api.routes import chair, health, model, reward_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting Ergonomic Chair AI API Service...")
    
    model_path = os.environ.get("CHAIR_MODEL_PATH", None)
    
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading model from: {model_path}")
        app.state.service = ChairAIService(model_path=model_path)
    else:
        logger.info("Starting without pre-loaded model (will use rule-based fallback)")
        app.state.service = ChairAIService()
    
    logger.info("Service initialized successfully")
    yield
    
    logger.info("Shutting down service...")


app = FastAPI(
    title="Ergonomic Chair AI Training API",
    description="""
## 人体工学座椅智能控制系统 API

基于强化学习 (PPO) 的具身智能系统，能够根据用户坐姿、压力分布和疲劳程度自动调整座椅参数。

### 主要功能

- **智能调整建议**: 根据实时传感器数据返回最优座椅调整方案
- **姿态分析**: 分析头部、肩部、骨盆姿态，识别不良坐姿
- **健康监测**: 监测静坐时长、疲劳度，提供健康提醒
- **批量处理**: 支持多时间点数据批量推理

### 快速开始

1. 访问 `/api/v1/chair/demo` 获取演示数据
2. 使用 `/api/v1/chair/adjust` (POST) 发送传感器数据获取调整建议
3. 查看 `/docs` 获取完整的交互式 API 文档

### 数据格式

**输入**: 8x8 压力矩阵 + 姿态角度 + 用户信息  
**输出**: 8维调整动作 + 置信度 + 舒适度评分 + 建议
    """,
    version="2.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加响应时间头"""
    start_time = datetime.datetime.now()
    response = await call_next(request)
    process_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "type": type(exc).__name__,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )


app.include_router(chair.router)
app.include_router(health.router)
app.include_router(model.router)
app.include_router(reward_config.router)

# WebSocket 路由（实时传感器数据接口）
from api.routes import websocket as ws_router
app.include_router(ws_router.router)


@app.get("/", tags=["Root"])
async def root():
    """根路径 - 返回 API 基本信息"""
    return {
        "service": "Ergonomic Chair AI Training System",
        "version": "2.2.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "endpoints": {
            "POST /api/v1/chair/adjust": "获取座椅调整建议 (REST)",
            "GET /api/v1/chair/demo": "演示接口 (REST)",
            "GET /api/v1/health": "健康检查 (REST)",
            "GET /api/v1/model": "模型信息 (REST)",
            "POST /api/v1/model/export": "导出 ONNX 模型 (REST)",
            "GET|PUT /api/v1/reward/config": "奖励函数配置管理 ⭐ [NEW]",
            "GET /api/v1/reward/presets": "奖励预设管理 ⭐ [NEW]",
            "POST /api/v1/reward/preview/calculate": "奖励预览计算 ⭐ [NEW]",
            "WS /ws/sensor": "实时传感器数据接口"
        },
        "reward_config_info": {
            "base_path": "/api/v1/reward",
            "features": [
                "动态参数配置",
                "6种内置预设 + 自定义预设",
                "实时预览与曲线绘制",
                "配置导入导出 (JSON/YAML)",
                "多配置对比分析",
                "配置评分与优化建议"
            ],
            "builtin_presets": [
                "balanced - 均衡模式",
                "health_first - 健康优先",
                "comfort_priority - 舒适优先",
                "strict_posture - 严格工效学",
                "energy_saving - 节能模式"
            ]
        },
        "websocket_info": {
            "endpoint": "/ws/sensor",
            "protocol": "WebSocket (JSON)",
            "features": [
                "实时双向通信",
                "心跳保活",
                "多用户会话管理",
                "异常主动推送"
            ],
            "message_types": {
                "client_to_server": ["sensor_data", "ping"],
                "server_to_client": ["adjustment", "alert", "pong", "error", "connected"]
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
