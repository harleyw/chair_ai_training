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
WebSocket 实时传感器数据接口
提供基于 WebSocket 的双向实时通信能力
"""

import json
import logging
import time
from typing import Any, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.ws_manager import manager as ws_manager
from api.service import ChairAIService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])


def get_service() -> ChairAIService:
    """获取服务实例"""
    from api.main import app
    return app.state.service


@router.websocket("/ws/sensor")
async def websocket_sensor_endpoint(websocket: WebSocket):
    """
    实时传感器数据 WebSocket 端点
    
    协议:
    - 客户端 → 服务端: sensor_data 消息（JSON）
    - 服务端 → 客户端: adjustment/alert/heartbeat/error 消息（JSON）
    
    消息格式:
    {
        "type": "sensor_data|ping",
        "timestamp": "ISO8601",
        "session_id": "uuid",
        "payload": { ... }
    }
    
    使用示例:
    ```javascript
    const ws = new WebSocket("ws://localhost:8000/ws/sensor");
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.type, data.payload);
    };
    ws.send(JSON.stringify({
        type: "sensor_data",
        payload: {
            pressure_matrix: [[...]],
            posture_angles: [15, -5, 10],
            sitting_duration: 1800,
            user_weight: 70,
            user_height: 1.70,
            fatigue_level: 0.3
        }
    }));
    ```
    """
    session_id = None
    service = get_service()
    
    try:
        # 1. 接受连接
        await websocket.accept()
        
        # 2. 注册会话
        session_id = await ws_manager.connect(websocket)
        
        # 3. 发送连接确认消息
        welcome_message = {
            "type": "connected",
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "session_id": session_id,
            "payload": {
                "message": "WebSocket connection established",
                "server_version": "1.0.1",
                "heartbeat_interval": ws_manager.heartbeat_interval,
                "supported_message_types": [
                    "sensor_data",
                    "ping"
                ]
            }
        }
        await ws_manager.send_personal_message(welcome_message, session_id)
        
        logger.info(f"WebSocket session started: {session_id}")
        
        # 4. 主消息循环
        while True:
            try:
                # 接收消息（带超时，用于心跳检测）
                data = await websocket.receive_text()
                
                start_time = time.time()
                
                # 解析 JSON
                try:
                    message = json.loads(data)
                except json.JSONDecodeError as e:
                    error_msg = {
                        "type": "error",
                        "payload": {
                            "error": "invalid_json",
                            "message": f"Invalid JSON: {str(e)}"
                        }
                    }
                    await ws_manager.send_personal_message(error_msg, session_id)
                    continue
                
                msg_type = message.get("type", "unknown")
                
                # 处理不同类型的消息
                if msg_type == "sensor_data":
                    await handle_sensor_data(
                        websocket, 
                        message, 
                        session_id, 
                        service,
                        start_time
                    )
                
                elif msg_type == "ping":
                    # 心跳响应
                    ws_manager.update_heartbeat(session_id)
                    
                    pong_msg = {
                        "type": "pong",
                        "timestamp": __import__('datetime').datetime.now().isoformat(),
                        "session_id": session_id,
                        "payload": {
                            "server_time": __import__('datetime').datetime.now().isoformat(),
                            "connections": len(ws_manager.active_connections)
                        }
                    }
                    await ws_manager.send_personal_message(pong_msg, session_id)
                
                else:
                    # 未知消息类型
                    warning_msg = {
                        "type": "warning",
                        "payload": {
                            "warning": "unknown_message_type",
                            "received_type": msg_type,
                            "supported_types": ["sensor_data", "ping"]
                        }
                    }
                    await ws_manager.send_personal_message(warning_msg, session_id)
                
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: {session_id}")
                break
                
            except Exception as e:
                logger.error(f"Error processing message for {session_id}: {e}")
                
                error_msg = {
                    "type": "error",
                    "payload": {
                        "error": "processing_error",
                        "message": str(e)
                    }
                }
                try:
                    await ws_manager.send_personal_message(error_msg, session_id)
                except:
                    break
    
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}")
    
    finally:
        # 清理连接
        if session_id:
            await ws_manager.disconnect(session_id)


async def handle_sensor_data(
    websocket: WebSocket,
    message: Dict[str, Any],
    session_id: str,
    service: ChairAIService,
    receive_time: float
):
    """
    处理传感器数据消息
    
    Args:
        websocket: WebSocket 连接
        message: 原始消息
        session_id: 会话 ID
        service: AI 服务实例
        receive_time: 消息接收时间戳
    """
    payload = message.get("payload", {})
    
    # 验证必要字段
    required_fields = ["pressure_matrix", "posture_angles", "sitting_duration"]
    missing_fields = [f for f in required_fields if f not in payload]
    
    if missing_fields:
        error_msg = {
            "type": "error",
            "payload": {
                "error": "missing_fields",
                "missing": missing_fields
            }
        }
        await ws_manager.send_personal_message(error_msg, session_id)
        return
    
    # 更新心跳
    ws_manager.update_heartbeat(session_id)
    
    # 调用 AI 服务进行处理
    try:
        action_vector, confidence = service.predict_action(payload)
        
        # 姿态分析
        analysis, issues, comfort_score, risk_level = service.analyze_posture(payload)
        
        # 计算处理延迟
        processing_latency = (time.time() - receive_time) * 1000
        
        # 构建调整建议响应
        response = {
            "type": "adjustment",
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "session_id": session_id,
            "payload": {
                "action": {
                    "seat_height": round(float(action_vector[0]), 4),
                    "backrest_angle": round(float(action_vector[1]), 4),
                    "lumbar_position": round(float(action_vector[2]), 4),
                    "lumbar_thickness": round(float(action_vector[3]), 4),
                    "headrest_position": round(float(action_vector[4]), 4),
                    "headrest_angle": round(float(action_vector[5]), 4),
                    "left_armrest": round(float(action_vector[6]), 4),
                    "right_armrest": round(float(action_vector[7]), 4)
                },
                "confidence": round(float(confidence), 4),
                "comfort_score": round(comfort_score, 2),
                "pressure_risk": risk_level,
                "posture_analysis": analysis,
                "posture_detail": analysis.get("posture_detail"),
                "recommendations": issues[:3] if issues else ["姿态良好"],
                "processing_latency_ms": round(processing_latency, 2)
            }
        }
        
        # 发送响应
        await ws_manager.send_personal_message(response, session_id)
        
        # 检查是否需要发送告警
        await check_and_send_alerts(session_id, payload, analysis, risk_level)
        
    except Exception as e:
        logger.error(f"Sensor processing error: {e}")
        
        error_response = {
            "type": "error",
            "payload": {
                "error": "processing_failed",
                "message": str(e)
            }
        }
        await ws_manager.send_personal_message(error_response, session_id)


async def check_and_send_alerts(
    session_id: str,
    sensor_data: Dict,
    analysis: Dict,
    risk_level: str
):
    """
    检查异常条件并发送告警
    
    Args:
        session_id: 会话 ID
        sensor_data: 传感器数据
        analysis: 姿态分析结果
        risk_level: 风险等级
    """
    alerts = []
    
    sitting_duration = sensor_data.get("sitting_duration", 0)
    fatigue_level = sensor_data.get("fatigue_level", 0)
    
    # 静坐超时告警 (>2小时)
    if sitting_duration > 7200:
        hours = int(sitting_duration // 3600)
        minutes = int((sitting_duration % 3600) // 60)
        alerts.append({
            "alert_type": "sitting_too_long",
            "severity": "critical" if sitting_duration > 10800 else "warning",
            "message": f"已连续静坐 {hours} 小时 {minutes} 分钟，请立即起身活动！",
            "recommendations": ["起身走动", "做伸展运动", "喝水休息"]
        })
    
    # 高疲劳度告警
    elif fatigue_level > 0.8 and risk_level in ["high", "medium"]:
        alerts.append({
            "alert_type": "fatigue_high",
            "severity": "warning",
            "message": "检测到高度疲劳状态，建议增加腰部支撑或短暂休息",
            "recommendations": ["调整腰托位置", "后仰靠背", "闭目养神 2 分钟"]
        })
    
    # 姿态严重偏差告警
    if analysis.get("head_posture") == "poor" or analysis.get("pelvis_posture") == "poor":
        alerts.append({
            "alert_type": "posture_warning",
            "severity": "warning",
            "message": "检测到不良坐姿，请及时调整",
            "recommendations": ["调整座椅高度", "挺直背部", "双脚平放地面"]
        })
    
    # 发送所有告警
    for alert in alerts:
        alert_message = {
            "type": "alert",
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "session_id": session_id,
            "payload": alert
        }
        await ws_manager.send_personal_message(alert_message, session_id)


@router.get("/ws/stats")
async def websocket_stats():
    """获取 WebSocket 连接统计信息"""
    stats = ws_manager.get_global_stats()
    return stats


@router.get("/ws/sessions/{session_id}")
async def get_session_info(session_id: str):
    """获取指定会话的详细信息"""
    session_stats = ws_manager.get_session_stats(session_id)
    
    if session_stats is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = ws_manager.get_session_history(session_id, last_n=20)
    
    return {
        **session_stats,
        "recent_messages": history
    }
