"""
SSE Server Transport Module

This module implements a Server-Sent Events (SSE) transport layer for MCP servers.

Example usage:
```
    # Create an SSE transport at an endpoint
    sse = SseServerTransport("/messages")

    # Create Starlette routes for SSE and message handling
    routes = [
        Route("/sse", endpoint=handle_sse),
        Route("/messages", endpoint=handle_messages, methods=["POST"])
    ]

    # Define handler functions
    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(
                streams[0], streams[1], app.create_initialization_options()
            )

    async def handle_messages(request):
        await sse.handle_post_message(request.scope, request.receive, request._send)

    # Create and run Starlette app
    starlette_app = Starlette(routes=routes)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
```

See SseServerTransport class documentation for more details.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional
from urllib.parse import quote
from uuid import UUID, uuid4
from dataclasses import dataclass

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import ValidationError
from sse_starlette import EventSourceResponse
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
from redis.asyncio import Redis
import pickle
import time

import mcp.types as types

logger = logging.getLogger(__name__)


@dataclass
class SerializableStreamWriter:
    """可序列化的流写入器包装类"""
    session_id: UUID
    created_at: float
    last_active: float
    
    @staticmethod
    def from_stream(session_id: UUID, writer: MemoryObjectSendStream) -> 'SerializableStreamWriter':
        """
        从流创建可序列化的包装对象
        
        Args:
            session_id: 会话ID
            writer: 原始的流写入器
            
        Returns:
            SerializableStreamWriter: 可序列化的包装对象
        """
        now = time.time()
        return SerializableStreamWriter(
            session_id=session_id,
            created_at=now,
            last_active=now
        )
    
    def is_expired(self, ttl: int) -> bool:
        """
        检查会话是否过期
        
        Args:
            ttl: 过期时间（秒）
            
        Returns:
            bool: 是否过期
        """
        return time.time() - self.last_active > ttl
    
    def update_activity(self) -> None:
        """更新最后活动时间"""
        self.last_active = time.time()


class SseServerTransport:
    """
    SSE server transport for MCP. This class provides _two_ ASGI applications,
    suitable to be used with a framework like Starlette and a server like Hypercorn:

        1. connect_sse() is an ASGI application which receives incoming GET requests,
           and sets up a new SSE stream to send server messages to the client.
        2. handle_post_message() is an ASGI application which receives incoming POST
           requests, which should contain client messages that link to a
           previously-established SSE session.
    """

    _endpoint: str
    _read_stream_writers: dict[
        UUID, MemoryObjectSendStream[types.JSONRPCMessage | Exception]
    ]

    def __init__(self, endpoint: str) -> None:
        """
        Creates a new SSE server transport, which will direct the client to POST
        messages to the relative or absolute URL given.
        """

        super().__init__()
        self._endpoint = endpoint
        self._read_stream_writers = {}
        logger.debug(f"SseServerTransport initialized with endpoint: {endpoint}")

    @asynccontextmanager
    async def connect_sse(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            logger.error("connect_sse received non-HTTP request")
            raise ValueError("connect_sse can only handle HTTP requests")

        read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
        read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]
        
        read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
        write_stream, write_stream_reader = anyio.create_memory_object_stream(0)
        
        session_id = uuid4()
        session_uri = f"{quote(self._endpoint)}?session_id={session_id.hex}"
        
        # 存储到 Redis
        await self._store_stream_writer(session_id, read_stream_writer)
        logger.debug(f"Successfully stored session {session_id}")

        # 创建 SSE 流
        sse_stream_writer, sse_stream_reader = anyio.create_memory_object_stream(
            0, dict[str, Any]
        )

        async def sse_writer():
            """处理 SSE 事件写入"""
            logger.debug("Starting SSE writer")
            async with sse_stream_writer, write_stream_reader:
                # 发送 endpoint 事件
                await sse_stream_writer.send({"event": "endpoint", "data": session_uri})
                logger.debug(f"Sent endpoint event: {session_uri}")

                # 转发消息
                async for message in write_stream_reader:
                    logger.debug(f"Sending message via SSE: {message}")
                    await sse_stream_writer.send(
                        {
                            "event": "message",
                            "data": message.model_dump_json(
                                by_alias=True, exclude_none=True
                            ),
                        }
                    )

        async def event_sender():
            """发送 SSE 事件到客户端"""
            response = EventSourceResponse(
                sse_stream_reader,
                ping=20,  # 发送 ping 事件的间隔（秒）
            )
            await response(scope, receive, send)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(sse_writer)
                tg.start_soon(event_sender)
                yield read_stream, write_stream
        finally:
            # 清理资源
            await self._redis.delete(f"mcp:session:{session_id}")
            self._read_stream_writers.pop(session_id, None)
            logger.debug(f"Cleaned up session {session_id}")

    async def handle_post_message(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        request = Request(scope, receive)
        session_id_param = request.query_params.get("session_id")
        
        if session_id_param is None:
            response = Response("session_id is required", status_code=400)
            return await response(scope, receive, send)
        
        try:
            session_id = UUID(hex=session_id_param)
            writer = await self._get_stream_writer(session_id)
            
            if not writer:
                response = Response("Session not found or expired", status_code=404)
                return await response(scope, receive, send)
            
            # 处理消息...
            json = await request.json()
            message = types.JSONRPCMessage.model_validate(json)
            await writer.send(message)
            
            # 刷新 session TTL
            await self._redis.expire(f"mcp:session:{session_id}", self._session_ttl)
            
            response = Response("Accepted", status_code=202)
            return await response(scope, receive, send)
            
        except ValueError:
            response = Response("Invalid session ID", status_code=400)
            return await response(scope, receive, send)


class DistributedSseServerTransport(SseServerTransport):
    def __init__(self, endpoint: str, redis_client: Redis) -> None:
        """
        初始化分布式 SSE 传输层
        
        Args:
            endpoint: SSE 端点
            redis_client: Redis 客户端实例
        """
        super().__init__(endpoint)
        self._redis = redis_client
        self._session_ttl = 3600  # 1小时过期
        
    async def _store_stream_writer(self, session_id: UUID, writer: MemoryObjectSendStream) -> None:
        try:
            # 存储可序列化的包装对象
            serializable_writer = SerializableStreamWriter.from_stream(session_id, writer)
            await self._redis.setex(
                f"mcp:session:{session_id}",
                self._session_ttl,
                pickle.dumps(serializable_writer)
            )
            # 在内存中保持原始的 writer 引用
            self._read_stream_writers[session_id] = writer
            logger.debug(f"Successfully stored session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store session in Redis: {e}")
            raise
    
    async def _get_stream_writer(self, session_id: UUID) -> Optional[MemoryObjectSendStream]:
        try:
            data = await self._redis.get(f"mcp:session:{session_id}")
            if data:
                serializable_writer = pickle.loads(data)
                # 更新最后活动时间
                serializable_writer.update_activity()
                await self._redis.setex(
                    f"mcp:session:{session_id}",
                    self._session_ttl,
                    pickle.dumps(serializable_writer)
                )
                # 返回内存中的原始 writer
                return self._read_stream_writers[session_id]
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve session from Redis: {e}")
            return None
            
    async def close(self) -> None:
        """关闭 Redis 连接"""
        await self._redis.close()
