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

from fastapi import APIRouter
from api.models import HealthStatus, ModelInfo
from api.service import ChairAIService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["系统"])


def get_service() -> ChairAIService:
    """获取服务实例"""
    from api.main import app
    return app.state.service


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="健康检查",
    description="检查 API 服务状态和模型加载情况"
)
async def health_check() -> HealthStatus:
    """服务健康检查端点"""
    service = get_service()
    
    return HealthStatus(
        status="healthy" if service else "unhealthy",
        service="Ergonomic Chair AI Training API",
        version="1.0.1",
        model_loaded=service.model_loaded if service else False,
        uptime_seconds=service.get_uptime() if service else 0,
        timestamp=__import__('datetime').datetime.now().isoformat()
    )


@router.get("/info", summary="服务信息", description="获取 API 服务基本信息")
async def service_info() -> dict:
    """返回服务基本信息"""
    return {
        "service": "Ergonomic Chair AI Training System",
        "version": "1.0.1",
        "description": "基于强化学习的人体工学座椅智能控制系统",
        "api_version": "v1",
        "endpoints": {
            "health": "/api/v1/health",
            "model_info": "/api/v1/model",
            "adjustment": "/api/v1/chair/adjust (POST)",
            "batch_adjustment": "/api/v1/chair/batch-adjust (POST)",
            "demo": "/api/v1/chair/demo",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "documentation": {
            "swagger_ui": "http://localhost:8000/docs",
            "redoc": "http://localhost:8000/redoc"
        }
    }
