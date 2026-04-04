from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Optional

import websockets
from websockets.server import ServerConnection

logger = logging.getLogger(__name__)


class WebSocketServer:
    """Async WebSocket broadcast server running in its own daemon thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._clients: set[ServerConnection] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[Any] = None
        self._ready = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-server")
        self._thread.start()
        self._ready.wait(timeout=5)
        logger.info("WebSocket server started on ws://%s:%s", self.host, self.port)

    def stop(self) -> None:
        if self._loop and self._server:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout=5)
        logger.info("WebSocket server stopped")

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── broadcast (thread-safe) ──────────────────────────────────────────

    def broadcast(self, data: dict) -> None:
        """Send a JSON message to all connected clients. Safe to call from any thread."""
        if not self._loop or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(data), self._loop)

    async def _broadcast_async(self, data: dict) -> None:
        msg = json.dumps(data, ensure_ascii=False)
        dead: list[ServerConnection] = []
        for ws in list(self._clients):
            try:
                await ws.send(msg)
            except websockets.ConnectionClosed:
                dead.append(ws)
            except Exception:
                logger.exception("Error sending to client")
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    # ── connection handler ───────────────────────────────────────────────

    async def _handler(self, websocket: ServerConnection) -> None:
        self._clients.add(websocket)
        logger.info("WS client connected (%d total)", len(self._clients))
        try:
            hello = {"event": "hello", "server": "transcription-app", "clients": len(self._clients)}
            await websocket.send(json.dumps(hello))
            async for _ in websocket:
                pass  # clients don't send meaningful messages yet
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("WS client disconnected (%d remaining)", len(self._clients))

    # ── internal ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())
        self._loop.close()

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        self._server = await websockets.serve(self._handler, self.host, self.port)
        self._ready.set()
        await self._stop_event.wait()
        self._server.close()
        await self._server.wait_closed()

    async def _shutdown(self) -> None:
        self._stop_event.set()
