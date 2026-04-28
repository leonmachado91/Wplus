"""Per-participant state and audio pipeline."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

import numpy as np

from app.audio.chunk_assembler import ChunkAssembler
from app.audio.vad_processor import VADProcessor
from app.multidevice.bleed_gate import BleedGateCoordinator, PendingChunk
from app.multidevice.opus_decoder import FRAME_SIZE, OpusStreamDecoder

if TYPE_CHECKING:
    from app.core.settings_manager import SettingsManager
    from app.transcription.groq_engine import TranscriptionEngine

logger = logging.getLogger(__name__)


@dataclass
class Participant:
    token: str
    display_name: str
    mode: str               # "presencial" | "remoto" | "auto"
    device_info: str
    joined_at: datetime
    muted: bool = False
    clock_offset_ms: float = 0.0
    # Stats updated in real time
    chunk_count: int = field(default=0, repr=False)


class ParticipantPipeline:
    """Complete audio pipeline for one participant.

    Data flow:
      WebSocket frame → OpusStreamDecoder (ffmpeg subprocess)
        → feeder thread (splits PCM into 512-sample frames)
          → VADProcessor (Silero, detects speech boundaries)
            → ChunkAssembler (PCM → WAV bytes, quality filters)
              → bridge thread (adds speaker meta, submits to Groq)
    """

    def __init__(
        self,
        participant: Participant,
        settings: "SettingsManager",
        engine: "TranscriptionEngine",
        session_start: datetime,
        gate_coordinator: Optional["BleedGateCoordinator"] = None,
    ) -> None:
        self.participant = participant
        self._engine = engine
        self._gate_coordinator = gate_coordinator
        # Offset to add to VAD-relative start_time so timeline reflects session time
        self._join_offset = (participant.joined_at - session_start).total_seconds()

        # Inter-stage queues
        self._raw_pcm_queue: queue.Queue = queue.Queue()
        self._speech_queue: queue.Queue = queue.Queue()
        self._transcription_queue: queue.Queue = queue.Queue()

        # Pipeline components
        vad_cfg = settings.get("vad")
        self._decoder = OpusStreamDecoder(sample_rate=16000)
        self._vad = VADProcessor(
            raw_pcm_queue=self._raw_pcm_queue,
            speech_queue=self._speech_queue,
            sample_rate=16000,
            onset_threshold=vad_cfg.get("onset_threshold", 0.5),
            offset_threshold=vad_cfg.get("offset_threshold", 0.35),
            onset_frames=vad_cfg.get("onset_frames", 2),
            offset_frames=vad_cfg.get("offset_frames", 6),
            min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 500),
            max_chunk_duration_s=vad_cfg.get("max_chunk_duration_s", 25),
            speech_pad_ms=vad_cfg.get("speech_pad_ms", 200),
        )
        self._chunk_asm = ChunkAssembler(
            speech_queue=self._speech_queue,
            transcription_queue=self._transcription_queue,
            sample_rate=16000,
        )

        self._stop_event = threading.Event()
        self._feeder_thread: Optional[threading.Thread] = None
        self._bridge_thread: Optional[threading.Thread] = None
        self._leftover_pcm: np.ndarray = np.array([], dtype=np.float32)
        # Latest client-side capture timestamp (ms, uint32 from frame header).
        # Updated on every frame; read by _bridge_loop() when building PendingChunk.
        self._last_client_ts_ms: float = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._decoder.start()
        self._vad.start()
        self._chunk_asm.start()
        token_short = self.participant.token[:8]
        self._feeder_thread = threading.Thread(
            target=self._feeder_loop,
            daemon=True,
            name=f"md-feeder-{token_short}",
        )
        self._bridge_thread = threading.Thread(
            target=self._bridge_loop,
            daemon=True,
            name=f"md-bridge-{token_short}",
        )
        self._feeder_thread.start()
        self._bridge_thread.start()
        logger.info(
            "Pipeline started for participant %s (join_offset=%.1fs)",
            self.participant.display_name, self._join_offset,
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._vad.stop()
        self._chunk_asm.stop()
        self._decoder.close()
        if self._feeder_thread:
            self._feeder_thread.join(timeout=2)
        if self._bridge_thread:
            self._bridge_thread.join(timeout=2)
        logger.info("Pipeline stopped for %s", self.participant.display_name)

    # ── frame ingestion ──────────────────────────────────────────────────

    def feed(self, webm_bytes: bytes, client_timestamp_ms: float = 0.0) -> None:
        """Called from the WebSocket handler when a new audio frame arrives.

        Args:
            webm_bytes: Raw WebM/Opus payload (after the 12-byte header).
            client_timestamp_ms: The uint32 timestamp embedded in the frame
                header by the client at capture time.  Used by BleedGate TDOA.
        """
        self._last_client_ts_ms = float(client_timestamp_ms)
        if not self.participant.muted:
            self._decoder.write(webm_bytes)

    # ── feeder thread ────────────────────────────────────────────────────

    def _feeder_loop(self) -> None:
        """Drains decoded PCM from ffmpeg and feeds 512-sample frames to VAD."""
        while not self._stop_event.is_set():
            time.sleep(0.02)  # 20 ms poll — fast enough given 200 ms WebM timeslice

            new_pcm = self._decoder.drain()
            if len(new_pcm) == 0:
                continue

            combined = (
                np.concatenate([self._leftover_pcm, new_pcm])
                if len(self._leftover_pcm) > 0
                else new_pcm
            )

            pos = 0
            while pos + FRAME_SIZE <= len(combined):
                self._raw_pcm_queue.put(combined[pos : pos + FRAME_SIZE].copy())
                pos += FRAME_SIZE
            self._leftover_pcm = combined[pos:].copy()

    # ── bridge thread ────────────────────────────────────────────────────

    def _bridge_loop(self) -> None:
        """Takes WAV chunks from ChunkAssembler and submits to TranscriptionEngine.

        When BleedGateCoordinator is active and enabled, chunks are forwarded to
        the gate instead of going directly to the engine. The gate's on_approved
        callback (wired in ParticipantManager) calls engine.submit() for approved
        chunks. When the gate is inactive (None or enabled=False), chunks bypass
        the gate entirely — same behaviour as before this change.
        """
        while not self._stop_event.is_set():
            try:
                wav_bytes, meta = self._transcription_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            meta = dict(meta)
            # Normalize to session-relative time
            meta["start_time"] = meta.get("start_time", 0.0) + self._join_offset
            # Speaker label — picked up by groq_engine._response_to_segment
            meta["provisional_speaker"] = self.participant.display_name
            self.participant.chunk_count += 1

            gate = self._gate_coordinator
            if gate is not None and gate.enabled:
                # Build a PendingChunk so the gate can compare RMS cross-participant
                import time as _time
                now_ms = _time.monotonic() * 1000.0
                deadline = now_ms + gate.window_ms + 100.0
                pending = PendingChunk(
                    token=self.participant.token,
                    speaker_name=self.participant.display_name,
                    mode=self.participant.mode,
                    started_at_ms=meta["start_time"] * 1000.0,
                    submit_time_ms=now_ms,
                    deadline_ms=deadline,
                    rms_mean=meta.get("rms_mean", 0.0),
                    wav_bytes=wav_bytes,
                    meta=meta,
                    client_timestamp_ms=self._last_client_ts_ms,
                )
                gate.submit(pending)
            else:
                self._engine.submit(wav_bytes, meta)
