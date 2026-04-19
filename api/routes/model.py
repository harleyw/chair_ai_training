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

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.models import ModelInfo, ExportResponse, ExportedList
from api.service import ChairAIService
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["模型管理"])


def get_service() -> ChairAIService:
    """获取服务实例"""
    from api.main import app
    return app.state.service


@router.get(
    "/model",
    response_model=ModelInfo,
    summary="获取模型信息",
    description="获取当前加载的 AI 模型的详细信息"
)
async def get_model_info() -> ModelInfo:
    """获取当前加载模型的信息"""
    service = get_service()
    
    info = service.get_model_info()
    
    return ModelInfo(**info)


@router.post("/model/load", summary="加载模型", description="加载指定路径的训练模型")
async def load_model(
    model_path: str = Query(..., description="模型文件路径 (.zip)"),
    description="加载新的训练模型到服务中"
) -> dict:
    """
    加载模型接口
    
    - **model_path**: 模型文件路径（支持相对或绝对路径）
    
    返回加载结果状态。
    """
    service = get_service()
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    
    success = service.load_model(model_path)
    
    if success:
        return {
            "success": True,
            "message": f"Model loaded successfully from {model_path}",
            "model_path": model_path,
            "loaded_at": service.load_time
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")


@router.get("/models", summary="列出可用模型", description="扫描 models 目录并列出所有可用的训练模型")
async def list_available_models() -> dict:
    """列出所有可用的模型文件"""
    model_dirs = ["./models", "./checkpoints", "../models"]
    found_models = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.zip'):
                    full_path = os.path.join(model_dir, f)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    modified = os.path.getmtime(full_path)
                    
                    found_models.append({
                        "name": f,
                        "path": full_path,
                        "size_mb": round(size_mb, 2),
                        "modified": __import__('datetime').datetime.fromtimestamp(modified).isoformat()
                    })
    
    return {
        "total_models": len(found_models),
        "models": sorted(found_models, key=lambda x: x['modified'], reverse=True),
        "message": "Use POST /api/v1/model/load?model_path=<path> to load a model" if found_models else "No models found. Train a model first using train.py"
    }


@router.get("/stats", summary="服务统计", description="获取推理次数、运行时间等统计信息")
async def get_stats() -> dict:
    """获取服务运行统计"""
    service = get_service()
    
    stats = service.get_stats()
    
    return {
        "service_status": "running",
        **stats,
        "uptime_formatted": _format_uptime(stats['uptime_seconds'])
    }


def _format_uptime(seconds: float) -> str:
    """格式化运行时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


@router.post("/model/export")
async def export_model_to_onnx(
    sb3_model_path: str = Query(..., description="Stable-Baselines3 模型路径 (.zip)"),
    output_path: str = Query(None, description="输出 ONNX 文件路径"),
    dynamic_batch: bool = Query(False, description="是否使用动态 batch 维度"),
    description="将 PPO 模型导出为 ONNX 格式"
):
    """导出模型为 ONNX 格式"""
    service = get_service()

    if not os.path.exists(sb3_model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {sb3_model_path}")

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(sb3_model_path))[0]
        suffix = "_dynamic" if dynamic_batch else ""
        output_path = f"{base_name}{suffix}.onnx"

    result = service.export_to_onnx(
        sb3_model_path=sb3_model_path,
        output_path=output_path,
        dynamic_batch=dynamic_batch
    )

    if result.get("success"):
        return ExportResponse(
            success=True,
            message="Model exported successfully",
            output_path=result.get("output_path"),
            file_size_mb=result.get("file_size_mb"),
            export_info=result
        )
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "Export failed"))


@router.get("/models/exported")
async def list_exported_models():
    """列出所有已导出的 ONNX 模型"""
    model_dirs = ["./models", "./exports", ".", "./onnx_models"]
    found_models = []

    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.onnx'):
                    full_path = os.path.join(model_dir, f)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    modified = os.path.getmtime(full_path)

                    found_models.append({
                        "name": f,
                        "path": full_path,
                        "size_mb": round(size_mb, 2),
                        "modified": __import__('datetime').datetime.fromtimestamp(modified).isoformat(),
                        "format": "onnx"
                    })

    return ExportedList(
        success=True,
        total_models=len(found_models),
        models=sorted(found_models, key=lambda x: x['modified'], reverse=True)
    )
