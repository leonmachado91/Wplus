import logging
import threading
import queue
from typing import Callable, Optional

from app.core.settings_manager import SettingsManager
from app.core.transcript_buffer import TranscriptBuffer
from app.audio.capture_engine import AudioCaptureEngine
from app.audio.vad_processor import VADProcessor
from app.audio.chunk_assembler import ChunkAssembler
from app.transcription.groq_engine import TranscriptionEngine

logger = logging.getLogger(__name__)


class ModeController:
    """Centralizes starting and stopping of application modes, completely decoupling logic from the UI."""

    def __init__(self, settings: SettingsManager, buffer: TranscriptBuffer):
        self._settings = settings
        self._buffer = buffer

        # Pipeline components
        self._capture: Optional[AudioCaptureEngine] = None
        self._vad: Optional[VADProcessor] = None
        self._chunk_asm: Optional[ChunkAssembler] = None
        self._engine: Optional[TranscriptionEngine] = None

        self._speech_queue: queue.Queue = queue.Queue()
        self._transcription_queue: queue.Queue = queue.Queue()
        self._bridge_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._is_live_running = False

        # Diarization (optional)
        self._diarization_engine = None
        self._speaker_mapper = None

        # External status listener — set by LivePanel to receive diarization messages
        self.on_diarization_status: Optional[Callable[[str], None]] = None

    # ── properties ───────────────────────────────────────────────────────

    @property
    def is_live_running(self) -> bool:
        return self._is_live_running

    @property
    def capture_engine(self) -> Optional[AudioCaptureEngine]:
        return self._capture

    @property
    def vad_processor(self) -> Optional[VADProcessor]:
        return self._vad

    @property
    def diarization_engine(self):
        return self._diarization_engine

    @property
    def speaker_mapper(self):
        return self._speaker_mapper

    # ── live mode ────────────────────────────────────────────────────────

    def start_mode_live(self, device_index: Optional[int], mode: str) -> None:
        if self._is_live_running:
            logger.warning("Live mode is already running.")
            return

        self._is_live_running = True
        self._stop_event.clear()

        # Fresh queues for the session
        self._speech_queue = queue.Queue()
        self._transcription_queue = queue.Queue()

        sample_rate = self._settings.get("audio", "sample_rate")
        vad_cfg = self._settings.get("vad")

        # Instantiate pipeline singletons for this session
        self._capture = AudioCaptureEngine(sample_rate=sample_rate)
        self._vad = VADProcessor(
            raw_pcm_queue=self._capture.raw_pcm_queue,
            speech_queue=self._speech_queue,
            sample_rate=sample_rate,
            onset_threshold=vad_cfg["onset_threshold"],
            offset_threshold=vad_cfg["offset_threshold"],
            min_speech_duration_ms=vad_cfg["min_speech_duration_ms"],
            max_chunk_duration_s=vad_cfg["max_chunk_duration_s"],
            speech_pad_ms=vad_cfg["speech_pad_ms"],
        )
        self._chunk_asm = ChunkAssembler(
            speech_queue=self._speech_queue,
            transcription_queue=self._transcription_queue,
            sample_rate=sample_rate,
        )
        self._engine = TranscriptionEngine(self._settings, self._buffer, sample_rate=sample_rate)

        # Start pipeline
        self._engine.start()
        self._vad.start()
        self._chunk_asm.start()
        self._capture.start(device_index=device_index, mode=mode)

        # Start diarization if enabled — reset cross-chunk speaker state for new session
        self._start_diarization_if_enabled()
        if self._diarization_engine:
            self._diarization_engine.reset_speakers()

        # Bridge queues → engine
        self._bridge_thread = threading.Thread(
            target=self._bridge_worker, daemon=True, name="mode-bridge-thread"
        )
        self._bridge_thread.start()

        logger.info(
            "ModeController: Live Mode 2 started (device=%s, mode=%s)", device_index, mode
        )

    def stop_mode_live(self) -> None:
        if not self._is_live_running:
            return

        self._stop_event.set()

        if self._capture:
            self._capture.stop()
        if self._chunk_asm:
            self._chunk_asm.stop()
        if self._vad:
            self._vad.stop()
        if self._engine:
            self._engine.stop()
        if self._diarization_engine:
            self._diarization_engine.stop()
            self._diarization_engine = None
        if self._speaker_mapper:
            self._speaker_mapper.reset()

        if self._bridge_thread:
            self._bridge_thread.join(timeout=2)
            self._bridge_thread = None

        self._is_live_running = False
        logger.info("ModeController: Live Mode 2 stopped.")

    # ── diarization ──────────────────────────────────────────────────────

    def _start_diarization_if_enabled(self) -> None:
        """Lazily start diarization engine if configured and enabled."""
        from app.core.settings_manager import SettingsManager  # already imported

        enabled = self._settings.get("diarization", "enabled")
        if not enabled:
            return

        hf_token = self._settings.get("api", "huggingface_token") or ""
        if not hf_token:
            logger.warning("Diarization enabled but no HuggingFace token configured — skipping.")
            return

        try:
            from app.diarization.diarization_engine import DiarizationEngine
            from app.diarization.speaker_mapper import SpeakerMapper

            custom_names = self._settings.get("diarization", "speaker_labels") or {}
            self._speaker_mapper = SpeakerMapper(custom_names=custom_names)
            
            sim_threshold = self._settings.get("diarization", "similarity_threshold")
            if sim_threshold is None:
                sim_threshold = 0.65

            self._diarization_engine = DiarizationEngine(
                hf_token=hf_token,
                on_result=self._on_diarization_result,
                on_status=self._on_diarization_status,
                use_gpu=True,
                similarity_threshold=float(sim_threshold),
            )
            self._diarization_engine.start()
            logger.info("Diarization engine started.")
        except Exception as e:
            logger.error("Failed to start diarization: %s", e)
            self._diarization_engine = None

    def _on_diarization_status(self, message: str) -> None:
        """Forward diarization status messages (model loading, errors) to any listener."""
        logger.info("Diarization status: %s", message)
        if self.on_diarization_status:
            try:
                self.on_diarization_status(message)
            except Exception:
                pass

    def refresh_speaker_mapper(self) -> None:
        """Re-apply custom speaker names from settings to the active SpeakerMapper."""
        if self._speaker_mapper:
            custom_names = self._settings.get("diarization", "speaker_labels") or {}
            self._speaker_mapper.update_custom_names(custom_names)

    def _on_diarization_result(self, segment_id: str, annotations: list[dict]) -> None:
        """Callback from DiarizationEngine — assigns dominant speaker to segment."""
        if not annotations:
            return

        # Pick the speaker with the longest duration in this chunk
        speaker_durations: dict[str, float] = {}
        for ann in annotations:
            spk = ann["speaker"]
            dur = ann["end"] - ann["start"]
            speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur

        dominant = max(speaker_durations, key=speaker_durations.get)  # type: ignore[arg-type]

        # Resolve display name for storage
        display = dominant
        if self._speaker_mapper:
            display = self._speaker_mapper.display_name(dominant)

        # Update segment in buffer (triggers UI refresh via signal)
        logger.info("Diarization callback: %s → %s (buffer update)", segment_id, display)
        updated = self._buffer.update_segment(segment_id, speaker=display)
        if updated:
            logger.info("Diarization applied: %s → speaker=%s", segment_id, display)
        else:
            logger.warning("Diarization: segment %s not found in buffer — may have been added after timeout", segment_id)

    # ── bridge ───────────────────────────────────────────────────────────

    def _bridge_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                wav_bytes, meta = self._transcription_queue.get(timeout=0.1)
                if self._engine:
                    self._engine.submit(wav_bytes, meta)

                # Send a copy to diarization in parallel
                if self._diarization_engine and self._diarization_engine.is_running:
                    seg_id = meta.get("segment_id", "")
                    self._diarization_engine.submit(wav_bytes, meta, segment_id=seg_id)

                self._transcription_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Bridge worker error: %s", e)
