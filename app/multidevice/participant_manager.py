"""Session lifecycle and participant registry for multi-device mode."""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, Optional

from app.multidevice.bleed_gate import BleedGateCoordinator, PendingChunk
from app.multidevice.participant import Participant, ParticipantPipeline
from app.multidevice.session_codes import generate_code

if TYPE_CHECKING:
    from app.core.settings_manager import SettingsManager
    from app.core.transcript_buffer import TranscriptBuffer
    from app.transcription.groq_engine import TranscriptionEngine

logger = logging.getLogger(__name__)

Listener = Callable[[str, dict], None]


class ParticipantManager:
    """Manages one active multi-device transcription session.

    Responsibilities:
    - Create / stop the session (code, start time)
    - Add / remove participants and their per-participant pipelines
    - Emit events to registered listeners (UI, REST broadcast)
    """

    def __init__(
        self,
        settings: "SettingsManager",
        engine: "TranscriptionEngine",
        buffer: "TranscriptBuffer",
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._buffer = buffer

        self._lock = threading.Lock()
        self._session_code: Optional[str] = None
        self._session_start: Optional[datetime] = None
        self._participants: Dict[str, Participant] = {}   # token → Participant
        self._pipelines: Dict[str, ParticipantPipeline] = {}  # token → Pipeline
        self._ws_connections: Dict[str, object] = {}          # token → WebSocket
        self._listeners: list[Listener] = []
        self._is_active = False

        # Bleed gate — shared across all participant pipelines in this session.
        # Disabled by default; user activates manually via set_gate_enabled(True).
        md_cfg = settings.get("multidevice") or {}
        self._gate = BleedGateCoordinator(
            window_ms=float(md_cfg.get("bleed_window_ms", 250)),
            margin_db=float(md_cfg.get("bleed_margin_db", 6.0)),
            enabled=bool(md_cfg.get("bleed_gate_enabled", False)),
        )
        # Approved chunks go directly to the shared transcription engine.
        self._gate.set_on_approved(self._on_chunk_approved)

    # ── listeners ────────────────────────────────────────────────────────

    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    def _notify(self, event: str, data: dict) -> None:
        for fn in self._listeners:
            try:
                fn(event, data)
            except Exception:
                logger.exception("Listener error on event %s", event)

    # ── session lifecycle ────────────────────────────────────────────────

    def create_session(self, session_name: str = "") -> str:
        with self._lock:
            if self._is_active:
                return self._session_code or ""
            self._session_code = generate_code()
            self._session_start = datetime.now()
            self._is_active = True
            code = self._session_code

        self._gate.start()
        self._buffer.start_session()
        self._notify("session_created", {"code": code, "name": session_name})
        logger.info("Multi-device session created: %s", code)
        return code

    def stop_session(self) -> None:
        with self._lock:
            if not self._is_active:
                return
            self._is_active = False
            tokens = list(self._participants.keys())

        # Stop all pipelines outside the lock (join can block)
        for token in tokens:
            self._stop_pipeline(token)

        self._gate.stop()

        with self._lock:
            self._participants.clear()
            self._pipelines.clear()
            self._ws_connections.clear()
            self._session_code = None
            self._session_start = None

        self._buffer.stop_session()
        self._notify("session_stopped", {})
        logger.info("Multi-device session stopped")

    # ── participant management ───────────────────────────────────────────

    def add_participant(
        self,
        token: str,
        display_name: str,
        mode: str = "auto",
        device_info: str = "",
    ) -> Optional[Participant]:
        # Handle reconnect: stop old pipeline outside lock first
        old_pipeline: Optional[ParticipantPipeline] = None
        with self._lock:
            if not self._is_active or self._session_start is None:
                return None
            if token in self._pipelines:
                old_pipeline = self._pipelines.pop(token)

            participant = Participant(
                token=token,
                display_name=display_name,
                mode=mode,
                device_info=device_info,
                joined_at=datetime.now(),
            )
            self._participants[token] = participant

            pipeline = ParticipantPipeline(
                participant=participant,
                settings=self._settings,
                engine=self._engine,
                session_start=self._session_start,
                gate_coordinator=self._gate,
            )
            self._pipelines[token] = pipeline

        if old_pipeline:
            old_pipeline.stop()

        pipeline.start()
        self._notify("participant_joined", {
            "token": token,
            "display_name": display_name,
            "mode": mode,
            "device_info": device_info,
        })
        logger.info("Participant joined: %s (%s)", display_name, token[:8])
        return participant

    def remove_participant(self, token: str) -> None:
        with self._lock:
            p = self._participants.pop(token, None)
            if p is None:
                return
            name = p.display_name

        self._stop_pipeline(token)

        with self._lock:
            self._pipelines.pop(token, None)
            self._ws_connections.pop(token, None)

        self._notify("participant_left", {"token": token, "display_name": name})
        logger.info("Participant left: %s (%s)", name, token[:8])

    def get_participant(self, token: str) -> Optional[Participant]:
        with self._lock:
            return self._participants.get(token)

    def get_pipeline(self, token: str) -> Optional[ParticipantPipeline]:
        with self._lock:
            return self._pipelines.get(token)

    def get_participants(self) -> list[Participant]:
        with self._lock:
            return list(self._participants.values())

    def rename_participant(self, token: str, new_name: str) -> None:
        with self._lock:
            p = self._participants.get(token)
            if p:
                p.display_name = new_name
        self._notify("participant_renamed", {"token": token, "display_name": new_name})

    def mute_participant(self, token: str, muted: bool) -> None:
        with self._lock:
            p = self._participants.get(token)
            if p:
                p.muted = muted
        self._notify("participant_muted", {"token": token, "muted": muted})

    # ── WebSocket registry ───────────────────────────────────────────────

    def register_ws(self, token: str, ws: object) -> None:
        with self._lock:
            self._ws_connections[token] = ws

    def unregister_ws(self, token: str) -> None:
        with self._lock:
            self._ws_connections.pop(token, None)

    # ── properties ───────────────────────────────────────────────────────

    @property
    def session_code(self) -> Optional[str]:
        return self._session_code

    @property
    def session_start(self) -> Optional[datetime]:
        return self._session_start

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def participant_count(self) -> int:
        with self._lock:
            return len(self._participants)

    # ── bleed gate control ───────────────────────────────────────────────

    def set_gate_enabled(self, enabled: bool) -> None:
        """Enable or disable the bleed gate at runtime (called from the UI)."""
        self._gate.enabled = enabled
        logger.info("BleedGate %s", "enabled" if enabled else "disabled")
        self._notify("gate_toggled", {"enabled": enabled})

    @property
    def gate_enabled(self) -> bool:
        return self._gate.enabled

    # ── internal ─────────────────────────────────────────────────────────

    def _on_chunk_approved(self, chunk: PendingChunk) -> None:
        """Callback fired by BleedGateCoordinator for each approved chunk."""
        self._engine.submit(chunk.wav_bytes, chunk.meta)

    def _stop_pipeline(self, token: str) -> None:
        with self._lock:
            pipeline = self._pipelines.get(token)
        if pipeline:
            pipeline.stop()
