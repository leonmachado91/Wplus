"""Chunk assembler — converts speech PCM segments to WAV bytes for transcription."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
from uuid import uuid4

from app.audio.audio_utils import pcm_to_wav_bytes

logger = logging.getLogger(__name__)

MIN_CHUNK_DURATION_MS = 300


class ChunkAssembler:
    """Receives speech audio from VADProcessor and produces WAV bytes for transcription.

    Consumes: (np.ndarray, chunk_meta) from speech_queue
    Produces: (bytes, chunk_meta) into transcription_queue
    """

    def __init__(
        self,
        speech_queue: queue.Queue,
        transcription_queue: queue.Queue,
        sample_rate: int = 16000,
    ) -> None:
        self._speech_queue = speech_queue
        self._transcription_queue = transcription_queue
        self.sample_rate = sample_rate

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="chunk-assembler")
        self._thread.start()
        logger.info("Chunk assembler started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Chunk assembler stopped")

    # ── processing ───────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio, meta = self._speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._process(audio, meta)

    def _process(self, audio: np.ndarray, meta: dict) -> None:
        duration_ms = meta.get("duration_ms", (len(audio) / self.sample_rate) * 1000)

        if duration_ms < MIN_CHUNK_DURATION_MS:
            logger.debug("Discarding chunk < %dms (was %.0fms)", MIN_CHUNK_DURATION_MS, duration_ms)
            return

        try:
            wav_bytes = pcm_to_wav_bytes(audio, self.sample_rate)
        except Exception:
            logger.exception("Failed to convert PCM to WAV")
            return

        # Pre-assign segment_id so Groq and Diarization share the same ID.
        # Groq will use it when creating TranscriptSegment; Diarization uses it
        # to call buffer.update_segment() after speaker assignment.
        meta = {**meta, "segment_id": f"seg-{uuid4().hex[:8]}"}

        logger.debug(
            "Chunk ready: %.1fs (%.0fms, %d bytes WAV, forced=%s, id=%s)",
            meta["start_time"],
            duration_ms,
            len(wav_bytes),
            meta.get("chunk_was_forced", False),
            meta["segment_id"],
        )

        self._transcription_queue.put((wav_bytes, meta))
