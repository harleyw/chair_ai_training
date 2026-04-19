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
WebSocket 连接管理器
管理所有活跃的 WebSocket 连接、会话状态和消息路由
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from collections import deque
import time

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket 连接管理器"""
    
    def __init__(
        self,
        max_connections: int = 100,
        heartbeat_interval: int = 30,
        session_timeout: int = 60,
        history_window_seconds: int = 300
    ):
        """
        初始化连接管理器
        
        Args:
            max_connections: 最大并发连接数
            heartbeat_interval: 心跳间隔（秒）
            session_timeout: 会话超时时间（秒）
            history_window_seconds: 历史数据窗口大小（秒）
        """
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        self.session_timeout = session_timeout
        self.history_window = history_window_seconds
        
        # 活跃连接映射: session_id -> WebSocket
        self.active_connections: Dict[str, Any] = {}
        
        # 会话状态: session_id -> session_data
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # 历史数据缓存: session_id -> deque of messages
        self.history: Dict[str, deque] = {}
        
        # 心跳追踪: session_id -> last_ping_time
        self.last_heartbeat: Dict[str, float] = {}
        
        # 统计信息
        self.total_connections = 0
        self.total_messages = 0
        
        # 后台任务引用
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: Any) -> str:
        """
        接受新连接并分配会话
        
        Args:
            websocket: WebSocket 实例
            
        Returns:
            session_id: 新会话的 UUID
            
        Raises:
            Exception: 当达到最大连接数限制时
        """
        if len(self.active_connections) >= self.max_connections:
            raise Exception(f"Maximum connections ({self.max_connections}) reached")
        
        session_id = str(uuid.uuid4())[:8]
        
        # 注册连接
        self.active_connections[session_id] = websocket
        
        # 初始化会话状态
        now = datetime.now().isoformat()
        self.sessions[session_id] = {
            "session_id": session_id,
            "connected_at": now,
            "last_activity": now,
            "message_count": 0,
            "user_info": {}
        }
        
        # 初始化历史数据窗口
        self.history[session_id] = deque(maxlen=1000)
        
        # 记录心跳时间
        self.last_heartbeat[session_id] = time.time()
        
        # 更新统计
        self.total_connections += 1
        
        logger.info(f"WebSocket connected: session={session_id}, total={len(self.active_connections)}")
        
        return session_id
    
    async def disconnect(self, session_id: str):
        """
        断开连接并清理资源
        
        Args:
            session_id: 会话 ID
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        if session_id in self.history:
            del self.history[session_id]
        
        if session_id in self.last_heartbeat:
            del self.last_heartbeat[session_id]
        
        logger.info(f"WebSocket disconnected: session={session_id}, remaining={len(self.active_connections)}")
    
    async def send_personal_message(
        self,
        message: Dict[str, Any],
        session_id: str
    ) -> bool:
        """
        向指定连接发送消息
        
        Args:
            message: 消息字典（将自动序列化为 JSON）
            session_id: 目标会话 ID
            
        Returns:
            是否发送成功
        """
        if session_id not in self.active_connections:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        try:
            websocket = self.active_connections[session_id]
            
            # 确保消息包含必要字段
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()
            if "session_id" not in message:
                message["session_id"] = session_id
            
            await websocket.send_json(message)
            
            # 更新统计
            self.total_messages += 1
            if session_id in self.sessions:
                self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
                self.sessions[session_id]["message_count"] += 1
            
            # 存入历史记录
            if session_id in self.history:
                self.history[session_id].append({
                    "type": message.get("type"),
                    "timestamp": message.get("timestamp"),
                    "data": message.get("payload", {})
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {session_id}: {e}")
            return False
    
    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude_session: Optional[str] = None
    ) -> int:
        """
        向所有连接广播消息
        
        Args:
            message: 消息字典
            exclude_session: 要排除的会话 ID（如发送者本身）
            
        Returns:
            成功发送的连接数
        """
        sent_count = 0
        
        for session_id in list(self.active_connections.keys()):
            if session_id == exclude_session:
                continue
            
            success = await self.send_personal_message(message, session_id)
            if success:
                sent_count += 1
        
        if sent_count > 0:
            logger.debug(f"Broadcast to {sent_count} connections")
        
        return sent_count
    
    def update_heartbeat(self, session_id: str):
        """
        更新心跳时间戳
        
        Args:
            session_id: 会话 ID
        """
        self.last_heartbeat[session_id] = time.time()
    
    def get_session_history(
        self,
        session_id: str,
        last_n: int = 50
    ) -> List[Dict]:
        """
        获取会话的历史消息
        
        Args:
            session_id: 会话 ID
            last_n: 返回最近 N 条消息
            
        Returns:
            消息列表
        """
        if session_id not in self.history:
            return []
        
        history_list = list(self.history[session_id])
        return history_list[-last_n:]
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """
        获取会话统计信息
        
        Args:
            session_id: 会话 ID
            
        Returns:
            统计信息字典或 None
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        history = self.history.get(session_id, deque())
        
        connected_at = datetime.fromisoformat(session["connected_at"])
        duration = (datetime.now() - connected_at).total_seconds()
        
        return {
            "session_id": session_id,
            "connected_at": session["connected_at"],
            "duration_seconds": round(duration, 1),
            "message_count": session["message_count"],
            "messages_per_second": round(session["message_count"] / max(duration, 0.001), 2),
            "history_size": len(history),
            "is_alive": session_id in self.active_connections
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        获取全局统计信息
        
        Returns:
            全局统计字典
        """
        active_sessions = [
            self.get_session_stats(sid) 
            for sid in self.active_connections.keys()
        ]
        
        return {
            "active_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            "total_connections_ever": self.total_connections,
            "total_messages": self.total_messages,
            "uptime_sessions": active_sessions,
            "server_time": datetime.now().isoformat()
        }
    
    async def check_timeouts(self):
        """
        检查并清理超时的连接（应在后台周期性调用）
        """
        current_time = time.time()
        timeout_sessions = []
        
        for session_id, last_ping in self.last_heartbeat.items():
            if current_time - last_ping > self.session_timeout:
                timeout_sessions.append(session_id)
        
        for session_id in timeout_sessions:
            logger.warning(f"Session timed out: {session_id}")
            
            # 尝试通知客户端（可能已断开）
            if session_id in self.active_connections:
                try:
                    await self.send_personal_message({
                        "type": "error",
                        "payload": {
                            "error": "session_timeout",
                            "message": "Session timed out due to inactivity"
                        }
                    }, session_id)
                except:
                    pass
                
            await self.disconnect(session_id)
    
    async def start_monitoring(self):
        """启动后台监控任务"""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Connection monitoring started")
    
    async def stop_monitoring(self):
        """停止后台监控任务"""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("Connection monitoring stopped")
    
    async def _monitor_loop(self):
        """后台监控循环：定期检查超时"""
        while True:
            try:
                await asyncio.sleep(10)  # 每10秒检查一次
                await self.check_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")


# 全局单例实例
manager = ConnectionManager()
