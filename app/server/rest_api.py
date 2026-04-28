from __future__ import annotations

import io
import logging
import socket
import threading
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.core.mode_controller import ModeController
from app.core.settings_manager import SettingsManager
from app.core.transcript_buffer import TranscriptBuffer

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _local_ip() -> str:
    """Best-effort: return the machine's primary LAN IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def create_app(settings: SettingsManager, buffer: TranscriptBuffer, mode_controller: Any = None) -> FastAPI:
    app = FastAPI(title="Transcription App API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],     # web client is same-origin; * needed for QR-linked phones
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Multi-device audio WebSocket router ──────────────────────────────
    from app.server.ws_audio_ingest import router as audio_ws_router
    app.include_router(audio_ws_router)

    # ── Multi-device web client routes ───────────────────────────────────

    @app.get("/join/{session_code}")
    def join_client(session_code: str) -> FileResponse:
        """Serve the participant web client HTML (session_code used in URL routing only)."""
        logger.debug("Serving client HTML for session %s", session_code)
        return FileResponse(_STATIC_DIR / "client.html", media_type="text/html")

    @app.get("/join/{session_code}/qr")
    def join_qr(session_code: str) -> Response:
        """Generate and return a QR code PNG for the session join URL."""
        try:
            import qrcode
        except ImportError:
            return Response(content=b"qrcode not installed", media_type="text/plain", status_code=500)

        port = settings.get("server", "rest_api_port")
        ip = _local_ip()
        url = f"http://{ip}:{port}/join/{session_code}"

        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    @app.get("/api/multidevice/status")
    def multidevice_status(request: Request) -> dict:
        """Return current multi-device session status."""
        manager = getattr(request.app.state, "participant_manager", None)
        if manager is None or not manager.is_active:
            return {"active": False}
        participants = [
            {
                "token": p.token[:8],
                "display_name": p.display_name,
                "mode": p.mode,
                "muted": p.muted,
                "chunk_count": p.chunk_count,
            }
            for p in manager.get_participants()
        ]
        return {
            "active": True,
            "session_code": manager.session_code,
            "participant_count": manager.participant_count,
            "participants": participants,
        }

    @app.get("/api/status")
    def get_status() -> dict:
        return {
            "status": "recording" if getattr(mode_controller, "is_live_running", False) else "idle",
            "session_id": buffer.session_id,
            "segment_count": len(buffer.get_segments()),
        }

    @app.get("/api/session/current")
    def get_session_current() -> dict:
        return {
            "is_recording": getattr(mode_controller, "is_live_running", False),
            "session_id": buffer.session_id
        }

    @app.post("/api/session/start")
    def start_session(body: dict[str, Any] = None) -> dict:
        body = body or {}
        mode = body.get("mode", "mic")
        device_index = body.get("device_index")

        if mode_controller and isinstance(mode_controller, ModeController):
            if not buffer.session_id:
                buffer.start_session()
            mode_controller.start_mode_live(device_index, mode)
            return {"ok": True, "message": "Recording started"}
        return {"ok": False, "error": "Mode controller not bound"}

    @app.post("/api/session/stop")
    def stop_session() -> dict:
        if mode_controller and isinstance(mode_controller, ModeController):
            mode_controller.stop_mode_live()
            return {"ok": True, "message": "Recording stopped"}
        return {"ok": False, "error": "Mode controller not bound"}

    @app.get("/api/settings")
    def get_settings() -> dict:
        return settings.to_safe_dict()

    @app.patch("/api/settings")
    def patch_settings(body: dict[str, Any]) -> dict:
        for section, values in body.items():
            if isinstance(values, dict):
                settings.update_section(section, values)
        return {"ok": True}

    @app.get("/api/devices")
    def get_devices() -> dict:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            result = []
            for i, d in enumerate(devices):
                result.append({
                    "index": i,
                    "name": d["name"],
                    "max_input_channels": d["max_input_channels"],
                    "max_output_channels": d["max_output_channels"],
                    "is_loopback": "loopback" in d["name"].lower(),
                })
            return {"devices": result}
        except Exception as e:
            return {"devices": [], "error": str(e)}

    @app.get("/api/transcript/current")
    def get_transcript() -> dict:
        segments = buffer.get_segments()
        return {
            "session_id": buffer.session_id,
            "segments": [s.to_dict() for s in segments],
        }

    @app.get("/api/transcript/current/text")
    def get_transcript_text() -> dict:
        return {
            "session_id": buffer.session_id,
            "text": buffer.get_plain_text(),
        }

    @app.delete("/api/transcript/current")
    def clear_transcript() -> dict:
        buffer.clear()
        return {"ok": True}

    # ── File Watcher (Mode 1) endpoints ──────────────────────────────────

    # NOTE: The file_watcher instance is lazily resolved from the app state.
    # It is set externally after the panel is created via app.state.file_watcher.

    @app.get("/api/watcher/status")
    def watcher_status() -> dict:
        fw = app.state.file_watcher if hasattr(app.state, "file_watcher") else None
        if not fw:
            return {"running": False, "jobs": []}
        return {
            "running": fw.is_running,
            "jobs": [j.to_dict() for j in fw.jobs],
        }

    @app.post("/api/watcher/start")
    def watcher_start(body: dict[str, Any] = None) -> dict:
        body = body or {}
        fw = app.state.file_watcher if hasattr(app.state, "file_watcher") else None
        if not fw:
            return {"ok": False, "error": "File watcher not initialized"}
        fw.start(
            watch_folder=body.get("watch_folder"),
            output_folder=body.get("output_folder"),
        )
        return {"ok": True, "message": "Watcher started"}

    @app.post("/api/watcher/stop")
    def watcher_stop() -> dict:
        fw = app.state.file_watcher if hasattr(app.state, "file_watcher") else None
        if not fw:
            return {"ok": False, "error": "File watcher not initialized"}
        fw.stop()
        return {"ok": True, "message": "Watcher stopped"}

    return app


class RESTAPIServer:
    """Runs FastAPI/uvicorn in a daemon thread."""

    def __init__(
        self,
        settings: SettingsManager,
        buffer: TranscriptBuffer,
        mode_controller: Any = None,
        host: str = "127.0.0.1",
        port: int = 8766,
    ) -> None:
        self.host = host
        self.port = port
        self._settings = settings
        self._buffer = buffer
        self._mode_controller = mode_controller
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._fastapi_app = create_app(self._settings, self._buffer, self._mode_controller)
        config = uvicorn.Config(
            self._fastapi_app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="rest-api")
        self._thread.start()
        logger.info("REST API started on http://%s:%s", self.host, self.port)

    @property
    def fastapi_app(self) -> Optional[FastAPI]:
        """The raw FastAPI application instance (not the ASGI middleware wrapper)."""
        return getattr(self, "_fastapi_app", None)

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        logger.info("REST API stopped")
