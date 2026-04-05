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
from app.transcription.segment import SpeakerSpan

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

        # ── floating mode (Mode 3) state ──────────────────────────────────
        self._float_capture: Optional[AudioCaptureEngine] = None
        self._float_vad: Optional[VADProcessor] = None
        self._float_chunk_asm: Optional[ChunkAssembler] = None
        self._float_engine: Optional[TranscriptionEngine] = None
        self._float_speech_queue: queue.Queue = queue.Queue()
        self._float_transcription_queue: queue.Queue = queue.Queue()
        self._float_bridge_thread: Optional[threading.Thread] = None
        self._float_stop_event = threading.Event()
        self._is_floating_running = False

    # ── properties ───────────────────────────────────────────────────────

    @property
    def is_live_running(self) -> bool:
        return self._is_live_running

    @property
    def is_floating_running(self) -> bool:
        return self._is_floating_running

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
            onset_frames=vad_cfg.get("onset_frames", 2),
            offset_frames=vad_cfg.get("offset_frames", 6),
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

        # P2: Reseta contexto dinâmico — evita que sessão anterior influencie esta
        self._engine.reset_context()

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

    # ── floating mode (Mode 3) ────────────────────────────────────────────

    def start_mode_floating(
        self,
        device_index: Optional[int],
        mode: str,
        on_segment: Callable,
    ) -> None:
        """Start the floating button pipeline.

        Reuses the same AudioCaptureEngine → VAD → ChunkAssembler → TranscriptionEngine
        chain as Mode 2, but delivers segments via *on_segment* callback instead of
        TranscriptBuffer. Diarization is never started in this mode.
        """
        if self._is_floating_running:
            logger.warning("Floating mode is already running.")
            return

        self._is_floating_running = True
        self._float_stop_event.clear()

        self._float_speech_queue = queue.Queue()
        self._float_transcription_queue = queue.Queue()

        sample_rate = self._settings.get("audio", "sample_rate")
        vad_cfg = self._settings.get("vad")

        self._float_capture = AudioCaptureEngine(sample_rate=sample_rate)
        self._float_vad = VADProcessor(
            raw_pcm_queue=self._float_capture.raw_pcm_queue,
            speech_queue=self._float_speech_queue,
            sample_rate=sample_rate,
            onset_threshold=vad_cfg["onset_threshold"],
            offset_threshold=vad_cfg["offset_threshold"],
            onset_frames=vad_cfg.get("onset_frames", 2),
            offset_frames=vad_cfg.get("offset_frames", 6),
            min_speech_duration_ms=vad_cfg["min_speech_duration_ms"],
            max_chunk_duration_s=vad_cfg["max_chunk_duration_s"],
            speech_pad_ms=vad_cfg["speech_pad_ms"],
        )
        self._float_chunk_asm = ChunkAssembler(
            speech_queue=self._float_speech_queue,
            transcription_queue=self._float_transcription_queue,
            sample_rate=sample_rate,
        )
        # buffer=None, on_segment delivers text straight to the injector
        self._float_engine = TranscriptionEngine(
            self._settings,
            buffer=None,
            sample_rate=sample_rate,
            on_segment=lambda seg: on_segment(seg.text),
        )

        self._float_engine.start()
        self._float_vad.start()
        self._float_chunk_asm.start()
        self._float_capture.start(device_index=device_index, mode=mode)

        self._float_bridge_thread = threading.Thread(
            target=self._float_bridge_worker,
            daemon=True,
            name="float-bridge-thread",
        )
        self._float_bridge_thread.start()

        logger.info(
            "ModeController: Floating Mode 3 started (device=%s, mode=%s)",
            device_index,
            mode,
        )

    def stop_mode_floating(self) -> None:
        if not self._is_floating_running:
            return

        self._float_stop_event.set()

        if self._float_capture:
            self._float_capture.stop()
        if self._float_chunk_asm:
            self._float_chunk_asm.stop()
        if self._float_vad:
            self._float_vad.stop()
        if self._float_engine:
            self._float_engine.stop()

        if self._float_bridge_thread:
            self._float_bridge_thread.join(timeout=2)
            self._float_bridge_thread = None

        self._is_floating_running = False
        logger.info("ModeController: Floating Mode 3 stopped.")

    @property
    def float_capture_engine(self) -> Optional[AudioCaptureEngine]:
        return self._float_capture

    @property
    def float_vad_processor(self) -> Optional[VADProcessor]:
        return self._float_vad

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
        """Callback from DiarizationEngine — assigns speaker(s) to segment.

        Single-speaker chunks (len == 1): behave exactly as before — set
        ``segment.speaker`` to the display name.

        Multi-speaker chunks (len > 1): cross-reference the time spans with the
        word timestamps already stored in the segment to produce ``sub_segments``.
        ``segment.speaker`` is still set to the dominant speaker so existing UI
        rendering continues to work without changes.
        """
        if not annotations:
            return

        # Resolve display names for all annotations
        resolved = []
        for ann in annotations:
            raw_id = ann["speaker"]
            display = self._speaker_mapper.display_name(raw_id) if self._speaker_mapper else raw_id
            resolved.append({**ann, "speaker": display})

        # Dominant speaker = longest total duration (backwards-compatible)
        speaker_durations: dict[str, float] = {}
        for ann in resolved:
            spk = ann["speaker"]
            dur = ann["end"] - ann["start"]
            speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
        dominant = max(speaker_durations, key=speaker_durations.get)  # type: ignore[arg-type]

        updates: dict = {"speaker": dominant}

        if len(resolved) > 1:
            # Multi-speaker: try to build sub_segments from word timestamps
            seg = self._buffer.get_segment(segment_id)
            if seg is not None and seg.words:
                sub_segs = self._assign_words_to_speakers(seg.words, resolved)
                if sub_segs:
                    updates["sub_segments"] = sub_segs
                    logger.info(
                        "Sub-chunk diarization: %s → %d speaker span(s)",
                        segment_id, len(sub_segs),
                    )
            else:
                # Words not yet available (Groq still pending) — store raw annotations
                # for a second pass once the segment arrives in the buffer.
                # For now we just log; the speaker label still gets applied via
                # _pending_speakers in TranscriptBuffer.
                logger.debug(
                    "Sub-chunk: segment %s words not available yet, skipping text split",
                    segment_id,
                )

        logger.info("Diarization callback: %s → speaker=%s", segment_id, dominant)
        updated = self._buffer.update_segment(segment_id, **updates)
        if updated:
            logger.info("Diarization applied: %s (dominant=%s)", segment_id, dominant)
        else:
            # A transcrição ainda não chegou ao buffer — o mecanismo de pending speaker
            # vai aplicar o speaker quando o Groq responder. Isso é esperado.
            logger.debug(
                "Diarization: segment %s ainda não no buffer (pending aplicado)",
                segment_id,
            )

    @staticmethod
    def _assign_words_to_speakers(
        words: list,
        annotations: list[dict],
    ) -> list[SpeakerSpan]:
        """Cross-reference word timestamps with speaker time spans.

        For each annotation span, collect words whose ``start`` timestamp falls
        within [span.start, span.end) and reconstruct the spoken text.
        Words that fall outside all spans (edge effects) are silently discarded.

        Args:
            words: list of WordTimestamp from the Groq response.
            annotations: list of {start, end, speaker} dicts (already display-name resolved).

        Returns:
            Ordered list of SpeakerSpan objects. Empty spans (no words) are dropped.
        """
        spans: list[SpeakerSpan] = []
        for ann in annotations:
            span_words = [
                w for w in words
                if ann["start"] <= w.start < ann["end"]
            ]
            if not span_words:
                continue
            text = " ".join(w.word for w in span_words).strip()
            if not text:
                # Words matched the span but all word strings are empty/whitespace.
                # Drop this span — the segment's full text will be used as fallback.
                continue
            spans.append(SpeakerSpan(
                speaker=ann["speaker"],
                start=ann["start"],
                end=ann["end"],
                text=text,
            ))
        return spans

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

    def _float_bridge_worker(self) -> None:
        """Bridge for the floating mode pipeline — no diarization."""
        while not self._float_stop_event.is_set():
            try:
                wav_bytes, meta = self._float_transcription_queue.get(timeout=0.1)
                if self._float_engine:
                    self._float_engine.submit(wav_bytes, meta)
                self._float_transcription_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Float bridge worker error: %s", e)
