from __future__ import annotations

import logging

from app.core.settings_manager import SettingsManager
from app.core.transcript_buffer import TranscriptBuffer
from app.server.rest_api import RESTAPIServer
from app.server.websocket_server import WebSocketServer

logger = logging.getLogger(__name__)


class ServerManager:
    """Starts and stops both WebSocket and REST servers."""

    def __init__(self, settings: SettingsManager, buffer: TranscriptBuffer, mode_controller: Any = None) -> None:
        self._settings = settings
        self._buffer = buffer
        self._mode_controller = mode_controller
        self.ws: WebSocketServer | None = None
        self.rest: RESTAPIServer | None = None

    def start(self) -> None:
        srv = self._settings.settings.server

        if srv.websocket_enabled:
            self.ws = WebSocketServer(host=srv.websocket_host, port=srv.websocket_port)
            self.ws.start()

        if srv.rest_api_enabled:
            self.rest = RESTAPIServer(
                self._settings,
                self._buffer,
                self._mode_controller,
                host="127.0.0.1",
                port=srv.rest_api_port,
            )
            self.rest.start()

        # wire buffer → ws broadcast
        if self.ws:
            self._buffer.add_listener(self._on_buffer_event)

        logger.info("ServerManager: all servers started")

    def stop(self) -> None:
        self._buffer.remove_listener(self._on_buffer_event)
        if self.rest:
            self.rest.stop()
        if self.ws:
            self.ws.stop()
        logger.info("ServerManager: all servers stopped")

    def _on_buffer_event(self, event: str, data: dict) -> None:
        if self.ws:
            self.ws.broadcast({"event": event, **data})
