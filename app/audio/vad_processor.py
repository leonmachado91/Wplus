"""Voice Activity Detection using Silero-VAD — state machine with pre-roll."""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VADProcessor:
    """Consumes raw PCM frames and emits speech segments.

    State machine: SILENCE ↔ SPEECH
      - onset:  prob > onset_threshold for `onset_frames` consecutive frames
      - offset: prob < offset_threshold for `offset_frames` consecutive frames
      - fallback: force-emit after max_chunk_duration_s of continuous speech
    """

    # Class-level cache: the model is heavy (∼8 MB JIT), loading it takes 0.5-1 s
    # and requires a GitHub network call.  Caching at the class level means the
    # model is loaded ONCE per Python process, not once per recording session.
    _model_cache: Optional["torch.jit.ScriptModule"] = None
    _model_cache_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        raw_pcm_queue: queue.Queue,
        speech_queue: queue.Queue,
        sample_rate: int = 16000,
        *,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.35,
        onset_frames: int = 2,
        offset_frames: int = 6,
        min_speech_duration_ms: int = 200,
        max_chunk_duration_s: int = 15,
        speech_pad_ms: int = 300,
    ) -> None:
        self._raw_queue = raw_pcm_queue
        self._speech_queue = speech_queue
        self.sample_rate = sample_rate

        # thresholds
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.onset_frames = onset_frames
        self.offset_frames = offset_frames
        self.min_speech_ms = min_speech_duration_ms
        self.max_chunk_s = max_chunk_duration_s
        self.speech_pad_ms = speech_pad_ms

        # state
        self._model: Optional[torch.jit.ScriptModule] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # pre-roll buffer: ~300ms of frames before speech onset
        frame_duration_ms = 32  # 512 samples / 16000 Hz ≈ 32ms
        pre_roll_frames = max(1, speech_pad_ms // frame_duration_ms)
        self._pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=pre_roll_frames)

        # VAD state
        self._is_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._speech_buffer: list[np.ndarray] = []
        self._speech_probs: list[float] = []
        self._speech_start_time: float = 0.0
        self._session_frame_count: int = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._load_model()
        self._stop_event.clear()
        self._reset_state()
        self._thread = threading.Thread(target=self._run, daemon=True, name="vad-processor")
        self._thread.start()
        logger.info("VAD processor started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        # flush any remaining speech
        if self._speech_buffer:
            self._emit_segment(forced=False)
        logger.info("VAD processor stopped")

    @property
    def is_speech(self) -> bool:
        return self._is_speech

    # ── model loading ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load Silero-VAD.
        
        We MUST load a fresh instance per thread. Silero-VAD is stateful and
        JIT-compiled PyTorch models cannot be safely shared or deep-copied across
        threads without state corruption (which causes random VAD triggers on noise).
        Loading from local disk cache takes ~0.15s, which is perfectly acceptable.
        """
        import torch
        logger.info("Loading Silero-VAD model for new participant...")
        t0 = time.monotonic()

        try:
            # First attempt: load with network check
            model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                trust_repo=True,
                skip_validation=True,
            )
        except Exception as net_err:
            logger.warning(
                "torch.hub error (%s: %s) — retrying from disk cache",
                type(net_err).__name__, net_err,
            )
            model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                trust_repo=True,
                skip_validation=True,
            )

        self._model = model
        self._model.reset_states()
        logger.info("Silero-VAD loaded in %.1fs", time.monotonic() - t0)

    # ── processing thread ────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> None:
        self._session_frame_count += 1

        # compute speech probability
        tensor = torch.from_numpy(frame).float()
        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()

        current_time = self._frame_to_seconds(self._session_frame_count)

        if not self._is_speech:
            # SILENCE state
            self._pre_roll.append(frame)

            if prob > self.onset_threshold:
                self._consecutive_speech += 1
                if self._consecutive_speech >= self.onset_frames:
                    # transition SILENCE → SPEECH
                    self._is_speech = True
                    self._consecutive_silence = 0
                    self._speech_start_time = current_time - (len(self._pre_roll) * 0.032)

                    # include pre-roll
                    self._speech_buffer = list(self._pre_roll)
                    self._speech_buffer.append(frame)
                    self._pre_roll.clear()
                    logger.debug("SPEECH onset at %.2fs (prob=%.3f)", self._speech_start_time, prob)
            else:
                self._consecutive_speech = 0
        else:
            # SPEECH state
            self._speech_buffer.append(frame)
            self._speech_probs.append(prob)

            if prob < self.offset_threshold:
                self._consecutive_silence += 1
                if self._consecutive_silence >= self.offset_frames:
                    # transition SPEECH → SILENCE
                    self._emit_segment(forced=False)
                    logger.debug("SPEECH offset at %.2fs (prob=%.3f)", current_time, prob)
            else:
                self._consecutive_silence = 0

            # force emit after max_chunk_duration_s
            speech_duration = current_time - self._speech_start_time
            if speech_duration >= self.max_chunk_s:
                self._emit_segment(forced=True)
                logger.debug("Forced chunk cut at %.2fs (%.1fs speech)", current_time, speech_duration)

    def _emit_segment(self, forced: bool) -> None:
        if not self._speech_buffer:
            return

        audio = np.concatenate(self._speech_buffer)
        duration_ms = (len(audio) / self.sample_rate) * 1000

        # discard very short segments
        if duration_ms < self.min_speech_ms:
            logger.debug("Discarding short segment: %.0fms", duration_ms)
            self._reset_state()
            return

        vad_confidence = float(np.mean(self._speech_probs)) if self._speech_probs else 0.0

        meta = {
            "start_time": self._speech_start_time,
            "end_time": self._speech_start_time + (len(audio) / self.sample_rate),
            "duration_ms": duration_ms,
            "chunk_was_forced": forced,
            "vad_confidence": vad_confidence,
        }

        self._speech_queue.put((audio, meta))
        self._reset_state()

    def _reset_state(self) -> None:
        self._is_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._speech_buffer = []
        self._speech_probs = []
        self._speech_start_time = 0.0

    def _frame_to_seconds(self, frame_count: int) -> float:
        """Convert frame index to seconds based on frame size."""
        from app.audio.capture_engine import FRAME_SIZE
        return (frame_count * FRAME_SIZE) / self.sample_rate

    def reset_session(self) -> None:
        """Reset session frame counter (call when starting a new recording)."""
        self._session_frame_count = 0
        self._reset_state()
        self._pre_roll.clear()
        if self._model is not None:
            self._model.reset_states()

    # ── quick evaluation ─────────────────────────────────────────────────

    @classmethod
    def evaluate_track(cls, audio_np: np.ndarray, sample_rate: int = 16000) -> float:
        """Avalia rapidamente a probabilidade de voz num chunk separado longo.
        
        Útil como 'VAD Secundário' para tracks que saíram do Conv-TasNet.
        Em vez de rodar frame a frame (32ms), tira a média global ou max-pool 
        de probabilidade se o sinal tem voz real ou é ruído residual fantasma.
        """
        if cls._model_cache is None:
            logger.warning("VAD não inicializado para evaluate_track.")
            return 0.0

        import torch
        
        # Silero VAD processa bem frames de 512 samples. 
        # Vamos tirar a media de de probabilidade na track inteira.
        tensor = torch.from_numpy(audio_np).float()
        
        frame_size = 512
        probs = []
        with torch.no_grad():
            for i in range(0, len(tensor), frame_size):
                chunk = tensor[i: i + frame_size]
                if len(chunk) < frame_size:
                    break
                prob = cls._model_cache(chunk, sample_rate).item()
                probs.append(prob)

        if not probs:
            return 0.0
            
        return sum(probs) / len(probs)
