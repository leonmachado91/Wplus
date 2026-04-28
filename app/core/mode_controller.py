import logging
import threading
import queue
from typing import Callable, Optional

from app.core.settings_manager import SettingsManager
from app.core.transcript_buffer import TranscriptBuffer
from app.audio.capture_engine import AudioCaptureEngine
from app.audio.vad_processor import VADProcessor
from app.audio.chunk_assembler import ChunkAssembler
from app.audio.audio_utils import pcm_to_wav_bytes
from app.transcription.groq_engine import TranscriptionEngine
from app.transcription.segment import SpeakerSpan
from uuid import uuid4
import numpy as np

logger = logging.getLogger(__name__)


class ModeController:
    """Centralizes starting and stopping of application modes, completely decoupling logic from the UI."""

    def __init__(self, settings: SettingsManager, buffer: TranscriptBuffer):
        self._settings = settings
        self._buffer = buffer

        # Lock that guards _is_live_running and _is_floating_running.
        # Prevents concurrent start/stop calls (e.g. from REST API + UI) from
        # launching two pipelines simultaneously or creating a torn-down state.
        self._pipeline_lock = threading.Lock()

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

        # Pending multi-speaker annotations: when diarization fires before Groq
        # responds, the segment isn't in the buffer yet and word assignment is
        # impossible.  We store the resolved annotation list here and retry as
        # soon as the segment_final event arrives from the buffer.
        self._pending_diarization: dict[str, list[dict]] = {}  # seg_id → annotations

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
        with self._pipeline_lock:
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
        # We no longer use ChunkAssembler for Live Mode. The bridge_worker processes the PCM directly 
        # using the new Source Separation Pipeline.
        self._chunk_asm = None
        self._engine = TranscriptionEngine(self._settings, self._buffer, sample_rate=sample_rate)

        # Load SeparatorEngine
        from app.diarization.separator_engine import SeparatorEngine
        sep_model_choice = self._settings.get("diarization", "separator_model") or "Conv-TasNet (Fast)"
        self._separator_engine = SeparatorEngine(
            use_gpu=True, 
            max_sources=2,
            model_type=sep_model_choice
        )
        if self._settings.get("diarization", "enable_source_separation"):
             threading.Thread(target=self._separator_engine.load_model, daemon=True).start()

        # P2: Reseta contexto dinâmico — evita que sessão anterior influencie esta
        self._engine.reset_context()

        # Start pipeline
        self._engine.start()
        self._vad.start()
        use_aec = self._settings.get("audio", "use_windows_aec")
        mic_normalize = self._settings.get("audio", "mic_normalize")
        self._capture.start(device_index=device_index, mode=mode, use_windows_aec=use_aec,
                            mic_normalize=bool(mic_normalize))

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
        with self._pipeline_lock:
            if not self._is_live_running:
                return
            # Mark as stopped early (inside lock) to prevent concurrent start attempts.
            self._is_live_running = False

        self._stop_event.set()

        if self._capture:
            self._capture.stop()
        # B5: Stop VAD before ChunkAssembler so the VAD's final flush is still
        # consumed by the assembler before it exits its own loop.
        if self._vad:
            self._vad.stop()
        if self._chunk_asm:
            self._chunk_asm.stop()
        if self._engine:
            self._engine.stop()
        if self._diarization_engine:
            self._diarization_engine.stop()
            self._diarization_engine = None
        if self._speaker_mapper:
            self._speaker_mapper.reset()

        # Clean up diarization retry state.
        try:
            self._buffer.remove_listener(self._on_buffer_for_diarization)
        except (ValueError, Exception):
            pass  # listener may not have been registered (e.g. diarization disabled)
        self._pending_diarization.clear()

        if self._bridge_thread:
            self._bridge_thread.join(timeout=2)
            self._bridge_thread = None

        logger.info("ModeController: Live Mode 2 stopped.")

    def set_paused(self, paused: bool) -> None:
        """Pause or resume live capture without stopping the stream."""
        if self._capture:
            self._capture.set_paused(paused)
        if self._float_capture:
            self._float_capture.set_paused(paused)



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
        with self._pipeline_lock:
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
        use_aec = self._settings.get("audio", "use_windows_aec")
        self._float_capture.start(device_index=device_index, mode=mode, use_windows_aec=use_aec)

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
        with self._pipeline_lock:
            if not self._is_floating_running:
                return
            self._is_floating_running = False

        self._float_stop_event.set()

        if self._float_capture:
            self._float_capture.stop()
        # B5: VAD before ChunkAssembler (same rationale as stop_mode_live)
        if self._float_vad:
            self._float_vad.stop()
        if self._float_chunk_asm:
            self._float_chunk_asm.stop()
        if self._float_engine:
            self._float_engine.stop()

        if self._float_bridge_thread:
            self._float_bridge_thread.join(timeout=2)
            self._float_bridge_thread = None

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
                sim_threshold = 0.35

            self._diarization_engine = DiarizationEngine(
                hf_token=hf_token,
                on_result=self._on_diarization_result,
                on_status=self._on_diarization_status,
                use_gpu=True,
                similarity_threshold=float(sim_threshold),
            )
            self._diarization_engine.start()
            logger.info("Diarization engine started.")

            # Register buffer listener for diarization retry.
            # When diarization finishes before Groq, we store the annotations and
            # apply them the moment the segment (with words) arrives.
            self._buffer.add_listener(self._on_buffer_for_diarization)
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
            # Multi-speaker: try to build sub_segments from word timestamps.
            seg = self._buffer.get_segment(segment_id)
            logger.debug(
                "Sub-chunk check: seg=%s found=%s words=%d",
                segment_id,
                seg is not None,
                len(seg.words) if seg else 0,
            )
            if seg is not None and seg.words:
                sub_segs = self._assign_words_to_speakers(seg.words, resolved, full_text=seg.text)
                logger.debug(
                    "Sub-chunk assign: %s → %d sub_segs (from %d words, %d spans, "
                    "word[0].start=%.2f word[0].word=%r ann[0].start=%.2f ann[0].end=%.2f)",
                    segment_id,
                    len(sub_segs),
                    len(seg.words),
                    len(resolved),
                    seg.words[0].start if seg.words else -1,
                    seg.words[0].word if seg.words else None,
                    resolved[0]["start"],
                    resolved[0]["end"],
                )
                if sub_segs:
                    # Se todos os spans tiverem o mesmo speaker, consolida em 1
                    # (evita que o scoring de boundary descarte a última palavra)
                    distinct_speakers = {s.speaker for s in sub_segs}
                    if len(distinct_speakers) == 1:
                        merged_text = " ".join(s.text for s in sub_segs).strip()
                        sub_segs = [SpeakerSpan(
                            speaker=sub_segs[0].speaker,
                            start=sub_segs[0].start,
                            end=sub_segs[-1].end,
                            text=merged_text,
                        )]

                    updates["sub_segments"] = sub_segs
                    logger.info(
                        "Sub-chunk diarization: %s → %d speaker span(s)",
                        segment_id, len(sub_segs),
                    )
            else:
                # Segment missing or words not yet available — store for retry.
                # _on_buffer_for_diarization will apply when segment_final fires.
                self._pending_diarization[segment_id] = resolved
                logger.debug(
                    "Diarization: %s stored as pending (seg_found=%s, words=%d)",
                    segment_id,
                    seg is not None,
                    len(seg.words) if seg else 0,
                )

        logger.info("Diarization callback: %s → speaker=%s", segment_id, dominant)
        updated = self._buffer.update_segment(segment_id, **updates)
        if updated:
            logger.info("Diarization applied: %s (dominant=%s)", segment_id, dominant)
        else:
            # Segment not yet in buffer (Groq still pending).
            # The buffer's _pending_speakers will apply the dominant speaker when
            # the segment arrives.  For multi-speaker chunks we also store the full
            # annotations so _on_buffer_for_diarization can retry word assignment.
            if len(resolved) > 1 and segment_id not in self._pending_diarization:
                self._pending_diarization[segment_id] = resolved
                logger.debug(
                    "Diarization: annotations stored as pending (not in buffer yet) for %s",
                    segment_id,
                )

    def _on_buffer_for_diarization(self, event: str, data: dict) -> None:
        """Buffer listener: retry word-level speaker assignment when segment arrives.

        When diarization completes before Groq returns, `_on_diarization_result`
        stores multi-speaker annotations in `_pending_diarization`.  Here we wait
        for the matching `segment_final` event (fired by the buffer once the Groq
        transcription is added), then cross the word timestamps with the stored
        speaker spans to produce `sub_segments`.
        """
        if event != "segment_final":
            return

        seg_id = data.get("segment", {}).get("id")
        if not seg_id or seg_id not in self._pending_diarization:
            return

        annotations = self._pending_diarization.pop(seg_id)
        seg = self._buffer.get_segment(seg_id)
        if seg is None or not seg.words:
            logger.debug(
                "Diarization retry: segment %s has no word timestamps — skipping sub-split",
                seg_id,
            )
            return

        sub_segs = self._assign_words_to_speakers(seg.words, annotations, full_text=seg.text)
        if sub_segs:
            # Se todos os spans tiverem o mesmo speaker, consolida em 1
            distinct_speakers = {s.speaker for s in sub_segs}
            if len(distinct_speakers) == 1:
                merged_text = " ".join(s.text for s in sub_segs).strip()
                sub_segs = [SpeakerSpan(
                    speaker=sub_segs[0].speaker,
                    start=sub_segs[0].start,
                    end=sub_segs[-1].end,
                    text=merged_text,
                )]

            self._buffer.update_segment(seg_id, sub_segments=sub_segs)
            logger.info(
                "Diarization retry applied: %s → %d speaker span(s)",
                seg_id, len(sub_segs),
            )
        else:
            logger.debug("Diarization retry: no valid word spans found for %s", seg_id)

    @staticmethod
    def _assign_words_to_speakers(
        words: list,
        annotations: list[dict],
        full_text: str = "",
    ) -> list[SpeakerSpan]:
        """Cross-reference word timestamps with speaker time spans.

        For each annotation span, collect words whose ``start`` timestamp falls
        within [span.start, span.end) and reconstruct the spoken text.

        When ``WordTimestamp.word`` is empty (older Groq SDK returns word objects
        without text content in some configurations), falls back to extracting the
        corresponding word slice from ``full_text`` by word index.

        Args:
            words: list of WordTimestamp from the Groq response.
            annotations: list of {start, end, speaker} dicts (already display-name resolved).
            full_text: the segment's full transcription text — used as text fallback.

        Returns:
            Ordered list of SpeakerSpan objects. Empty spans (no words) are dropped.
        """
        spans: list[SpeakerSpan] = []
        full_text_words = full_text.split() if full_text else []
        prev_cut_idx = 0  # rastreia onde o span anterior terminou (por índice)

        for j, ann in enumerate(annotations):
            ann_end = ann["end"]

            if j == len(annotations) - 1:
                # Último span: pega TODAS as palavras restantes (nenhuma fica de fora)
                indices = list(range(prev_cut_idx, len(words)))
            else:
                # Encontra o índice de corte mais inteligente para este boundary
                best_cut_idx = len(words)  # fallback: tudo vai pro próximo span
                best_score = float("inf")

                for i in range(prev_cut_idx, len(words)):
                    w = words[i]
                    w_mid = (w.start + w.end) / 2.0

                    # Considera apenas cortes próximos ao boundary temporal (±1.5s)
                    time_diff = abs(w_mid - ann_end)
                    if time_diff > 1.5:
                        continue

                    # Bônus se a palavra ANTERIOR termina com pontuação forte
                    bonus = 0.0
                    if i > 0:
                        prev_word = words[i - 1].word.strip()
                        if prev_word.endswith(('.', '!', '?')):
                            bonus = 1.0
                        elif prev_word.endswith((',', ';', ':')):
                            bonus = 0.4

                    score = time_diff - bonus
                    if score < best_score:
                        best_score = score
                        best_cut_idx = i

                # Palavras deste span: do índice anterior até best_cut_idx (exclusivo)
                indices = list(range(prev_cut_idx, best_cut_idx))
                prev_cut_idx = best_cut_idx  # próximo span começa daqui

            # Try word-level text from WordTimestamp.word (populated by groq_engine)
            text = " ".join(words[i].word for i in indices).strip()

            if not text and full_text_words:
                # Fallback: .word was empty (SDK returned typed objects without text).
                # Reconstruct by extracting the matching slice from full_text.
                text = " ".join(
                    full_text_words[i] for i in indices if i < len(full_text_words)
                ).strip()

            if not text:
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
        """Consumes PCM directly from VAD, applies Source Separation, identifies speakers,
        and submits all valid tracks concurrently to TranscriptionEngine.
        """
        sample_rate = self._settings.get("audio", "sample_rate")
        
        while not self._stop_event.is_set():
            try:
                audio_np, meta = self._speech_queue.get(timeout=0.1)
                
                # Check minimum duration from VAD
                duration_ms = meta.get("duration_ms", (len(audio_np) / sample_rate) * 1000)
                if duration_ms < 300:
                    continue
                    
                # 1. Source Separation Boolean Gate
                overlap_detected = False
                if getattr(self, "_diarization_engine", None) and getattr(self, "_separator_engine", None) and self._settings.get("diarization", "enable_source_separation"):
                    # SettingsManager accepts exactly 2 arguments for key lookup
                    overlap_thresh_val = self._settings.get("diarization", "overlap_threshold")
                    if overlap_thresh_val is None:
                        overlap_thresh_val = 0.15
                    overlap_detected = self._diarization_engine.has_overlapping_speakers(audio_np, sample_rate, threshold=float(overlap_thresh_val))
                    
                if overlap_detected and getattr(self, "_separator_engine", None) and self._separator_engine.is_ready:
                    tracks = self._separator_engine.separate(audio_np, sample_rate)
                else:
                    tracks = [audio_np]

                valid_tracks = []
                
                # Global RMS Filter (Relational Acoustics)
                rms_list = [float(np.sqrt(np.mean(t ** 2))) for t in tracks]
                max_rms = max(rms_list) if rms_list else 0.0

                crosstalk_ratio = self._settings.get("diarization", "crosstalk_filter_ratio")
                if crosstalk_ratio is None:
                    crosstalk_ratio = 0.30

                for track_audio, rms in zip(tracks, rms_list):
                    # Descartamos a trilha APENAS se ela for, na MÉDIA GERAL do chunk,
                    # absurdamente mais silenciosa que a trilha principal.
                    if rms < max(0.005, max_rms * float(crosstalk_ratio)): 
                        continue
                        
                    vad_prob = VADProcessor.evaluate_track(track_audio, sample_rate)
                    if vad_prob < 0.45:
                        continue
                        
                    # 3. Speaker Identification (Soft-Match)
                    if self._diarization_engine and self._diarization_engine.is_running:
                        candidates = self._diarization_engine.identify_speakers_sync(track_audio, sample_rate)
                    else:
                        candidates = []

                    # 4. Prepare WAV and Meta
                    wav_bytes = pcm_to_wav_bytes(track_audio, sample_rate)
                    track_meta = meta.copy()
                    track_meta["segment_id"] = f"seg-{uuid4().hex[:8]}"
                    if candidates:
                        track_meta["speaker_candidates"] = candidates
                        # Preenche a label provisória (usada se o Groq aprovar no pós-filtro)
                        best_spk = candidates[0][0]
                        if self._speaker_mapper:
                            best_spk = self._speaker_mapper.display_name(best_spk)
                        track_meta["provisional_speaker"] = best_spk
                    
                    valid_tracks.append((wav_bytes, track_meta))

                # 5. Submit for concurrent Groq transcription + Levenshtein Deduplication
                if self._engine and valid_tracks:
                    # Notice we don't send one by one anymore, we expect the engine 
                    # to resolve overlapping texts for this same timestamp event.
                    self._engine.submit_parallel(valid_tracks)

                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Bridge worker error: %s", e, exc_info=True)

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
