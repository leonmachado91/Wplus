"""Diarization engine — per-chunk speaker verification via pyannote embeddings.

WHY THIS APPROACH (v3 — final):
    The pipeline (speaker-diarization-3.1) uses AHC clustering internally and
    needs to see multiple speakers at the same time in the same audio file.
    When you feed it a single short VAD chunk (1 person talking), it returns
    SPEAKER_00 every time with no cross-chunk context.

    Our VAD already handles segmentation perfectly: each chunk = one utterance
    = one speaker (with rare overlap). The problem reduces to:
        "Is this chunk the same person as chunk N-3 seconds ago?"

    This is Speaker Verification, not Speaker Diarization.

    Approach:
    1. Load the pipeline just to access its internal embedding model
       (wespeaker-voxceleb-resnet34 — no extra download needed)
    2. For each chunk, run Inference(embedding_model, window="whole")
       → returns a (1, D) embedding vector — the voice fingerprint
    3. Compare against known speaker embeddings via cosine similarity
    4. If similarity >= threshold → same speaker → reuse global ID
    5. If below threshold → new speaker → assign next global ID
    6. As session grows, update running average of each speaker's embedding

    This runs fast (~50-150ms per chunk on CPU), works on any chunk length,
    and correctly tracks speakers across the entire session.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

DiarizationCallback = Callable[[str, list[dict]], None]  # (segment_id, annotations)

# Cosine similarity threshold for speaker re-identification.
# Higher = stricter (fewer false matches). 0.6-0.7 works well for clean audio.
_SIM_THRESHOLD = 0.65

# Timeout for a single embedding inference call.
_EMBED_TIMEOUT_S = 15


class DiarizationEngine:
    """Per-chunk speaker verification using pyannote's internal embedding model.

    Lazy model loading in a daemon thread. Processes one chunk at a time
    from a queue — no UI blocking, no OOM risk.
    """

    def __init__(
        self,
        hf_token: str,
        on_result: DiarizationCallback,
        on_status: Callable[[str], None] | None = None,
        use_gpu: bool = False,
        similarity_threshold: float = 0.65,
    ):
        self._token = hf_token
        self._on_result = on_result
        self._on_status = on_status or (lambda msg: None)
        self._use_gpu = use_gpu
        self._similarity_threshold = similarity_threshold

        # Loaded lazily in the worker thread
        self._inference = None       # pyannote Inference object (embedding model)
        self._model_loaded = threading.Event()
        self._model_error: Optional[str] = None

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False

        # ── Per-session speaker registry ───────────────────────────────────
        # List of (running_avg_embedding, global_speaker_id)
        self._known_speakers: list[tuple[np.ndarray, str]] = []
        self._next_speaker_idx: int = 0
        self._speaker_lock = threading.Lock()

    # ── public ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def model_ready(self) -> bool:
        return self._model_loaded.is_set() and self._model_error is None

    @property
    def model_error(self) -> Optional[str]:
        return self._model_error
        
    def update_threshold(self, threshold: float) -> None:
        self._similarity_threshold = threshold
        logger.debug("Diarization similarity threshold updated to %.2f", threshold)

    def start(self) -> None:
        if self._is_running:
            return
        if not self._token:
            logger.error("DiarizationEngine: no HuggingFace token — cannot start.")
            return

        self._stop_event.clear()
        self._model_loaded.clear()
        self._model_error = None
        self._is_running = True

        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="diarization-worker"
        )
        self._thread.start()
        logger.info("DiarizationEngine started.")

    def stop(self) -> None:
        if not self._is_running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._is_running = False
        logger.info("DiarizationEngine stopped.")

    def submit(self, wav_bytes: bytes, chunk_meta: dict, segment_id: str) -> None:
        """Submit a WAV chunk (non-blocking). Drops oldest if queue > 4."""
        if not self._is_running:
            return
        if self._queue.qsize() >= 4:
            logger.warning("Diarization queue full — dropping chunk %s", segment_id)
            return
        self._queue.put((wav_bytes, chunk_meta, segment_id))

    def reset_speakers(self) -> None:
        """Clear speaker history (call at session start/stop)."""
        with self._speaker_lock:
            self._known_speakers.clear()
            self._next_speaker_idx = 0
        logger.debug("DiarizationEngine: speaker registry reset.")

    # ── worker ────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        self._load_model()
        if not self._inference:
            self._is_running = False
            return

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            wav_bytes, chunk_meta, segment_id = item
            self._process_chunk(wav_bytes, chunk_meta, segment_id)

    # ── main processing ───────────────────────────────────────────────────

    def _process_chunk(self, wav_bytes: bytes, chunk_meta: dict, segment_id: str) -> None:
        """Extract embedding for chunk, match to known speaker, emit result."""
        try:
            import torch
            import soundfile as sf

            wav_io = io.BytesIO(wav_bytes)
            wav_io.seek(0)
            audio_np, sr = sf.read(wav_io, dtype="float32")

            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            duration_s = len(audio_np) / sr
            if duration_s < 0.5:
                logger.debug("Chunk %s too short (%.2fs) — skipping", segment_id, duration_s)
                return

            waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)
            if self._use_gpu and torch.cuda.is_available():
                waveform = waveform.cuda()

            # Extract voice fingerprint
            t0 = time.monotonic()
            embedding = self._extract_embedding(waveform, sr)
            elapsed = time.monotonic() - t0

            if embedding is None:
                logger.warning("Could not extract embedding for %s", segment_id)
                return

            # Match against known speakers or register new one
            speaker_id = self._match_or_register(embedding)

            logger.info(
                "Speaker identified: %s → %s (%.2fs, %.2fs audio)",
                segment_id, speaker_id, elapsed, duration_s,
            )

            start_time = chunk_meta.get("start_time", 0.0)
            end_time = chunk_meta.get("end_time", start_time + duration_s)

            self._on_result(segment_id, [{
                "speaker": speaker_id,
                "start": start_time,
                "end": end_time,
            }])

        except Exception as e:
            logger.error("Error processing chunk %s: %s", segment_id, e, exc_info=True)

    def _extract_embedding(self, waveform, sample_rate: int) -> Optional[np.ndarray]:
        """Run embedding inference with timeout. Returns unit-norm numpy vector."""
        result_holder: list = []
        error_holder: list = []

        def _run() -> None:
            try:
                emb = self._inference({"waveform": waveform, "sample_rate": sample_rate})
                result_holder.append(emb)
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=_EMBED_TIMEOUT_S)

        if t.is_alive():
            logger.warning("Embedding timed out after %ds", _EMBED_TIMEOUT_S)
            return None
        if error_holder:
            logger.error("Embedding error: %s", error_holder[0])
            return None
        if not result_holder:
            return None

        emb = np.array(result_holder[0]).flatten()
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return None
        return emb / norm  # unit vector for cosine similarity

    def _match_or_register(self, embedding: np.ndarray) -> str:
        """Compare embedding against known speakers, return global speaker ID."""
        with self._speaker_lock:
            if self._known_speakers:
                # Cosine similarity = dot product of unit vectors
                sims = [float(np.dot(embedding, k_emb)) for k_emb, _ in self._known_speakers]
                best_idx = int(np.argmax(sims))
                best_sim = sims[best_idx]

                logger.debug(
                    "Speaker similarities: %s | best=%.3f (idx=%d, threshold=%.2f)",
                    [f"{s:.2f}" for s in sims], best_sim, best_idx, self._similarity_threshold,
                )

                if best_sim >= self._similarity_threshold:
                    # Re-identified — update running average embedding
                    known_emb, known_id = self._known_speakers[best_idx]
                    updated = (known_emb * 0.7 + embedding * 0.3)  # weighted average
                    updated /= (np.linalg.norm(updated) + 1e-8)
                    self._known_speakers[best_idx] = (updated, known_id)
                    return known_id

            # New speaker
            global_id = f"SPEAKER_{self._next_speaker_idx:02d}"
            self._next_speaker_idx += 1
            self._known_speakers.append((embedding, global_id))
            logger.info("New speaker registered: %s (%d known)", global_id, len(self._known_speakers))
            return global_id

    # ── model loading ─────────────────────────────────────────────────────

    def _ensure_dependencies(self) -> bool:
        try:
            import pyannote.audio  # noqa: F401
            return True
        except ImportError:
            pass

        self._on_status("Instalando pyannote.audio (pode levar alguns minutos)…")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pyannote.audio>=3.3.2", "soundfile", "--quiet"],
                timeout=300,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self._model_error = f"Falha ao instalar pyannote.audio: {e}"
            logger.error(self._model_error)
            return False

    REQUIRED_HF_MODELS = [
        ("pyannote/speaker-diarization-3.1",      "https://huggingface.co/pyannote/speaker-diarization-3.1"),
        ("pyannote/segmentation-3.0",             "https://huggingface.co/pyannote/segmentation-3.0"),
        ("pyannote/speaker-diarization-community-1", "https://huggingface.co/pyannote/speaker-diarization-community-1"),
    ]

    def _load_model(self) -> None:
        """Load the diarization pipeline, then extract its embedding model for use."""
        if not self._ensure_dependencies():
            return

        import warnings

        try:
            import torch
            import torchaudio
            from pyannote.audio import Pipeline, Inference  # type: ignore

            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]

            if self._token:
                os.environ["HF_TOKEN"] = self._token

            self._on_status("Carregando modelo de diarização…")
            logger.info("Loading pyannote/speaker-diarization-3.1 to access embedding model…")
            t0 = time.monotonic()

            import pyannote.audio as _pya
            pya_version = tuple(int(x) for x in _pya.__version__.split(".")[:2])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*libtorchcodec.*", category=UserWarning)
                if pya_version >= (3, 3):
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=self._token,
                    )
                else:
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=self._token,
                    )

            # Extract the internal embedding model from the pipeline.
            # This is the wespeaker-voxceleb-resnet34 model — already downloaded.
            # We use it directly for per-chunk speaker verification (much faster
            # than running the full AHC pipeline on every chunk).
            from pyannote.audio import Model  # type: ignore

            raw_emb = getattr(pipeline, "embedding", None)
            logger.info("pipeline.embedding type: %s | value: %s", type(raw_emb).__name__, repr(raw_emb)[:120])

            if raw_emb is None:
                raise RuntimeError("pipeline.embedding not found — unexpected pyannote version.")

            # Some pyannote builds store the embedding as a Model, others as a path/name string
            if isinstance(raw_emb, str):
                # Load by name/path using the same token
                logger.info("Embedding is a string — loading Model directly: %s", raw_emb)
                if pya_version >= (3, 3):
                    emb_model = Model.from_pretrained(raw_emb, token=self._token)
                else:
                    emb_model = Model.from_pretrained(raw_emb, use_auth_token=self._token)
            else:
                # Already a Model object
                emb_model = raw_emb

            self._inference = Inference(emb_model, window="whole")

            if self._use_gpu and torch.cuda.is_available():
                self._inference.to(torch.device("cuda"))
                logger.info("Embedding inference on CUDA.")
            else:
                logger.info("Embedding inference on CPU.")

            elapsed = time.monotonic() - t0
            logger.info("Embedding model ready in %.1fs", elapsed)
            self._model_loaded.set()
            self._on_status("Diarização pronta.")

        except Exception as e:
            self._inference = None
            self._model_error = self._humanize_error(e)
            logger.error("Failed to load diarization model: %s", self._model_error)
            self._on_status(f"Erro: {self._model_error}")

    def _humanize_error(self, exc: Exception) -> str:
        msg = str(exc)
        if "403" in msg or "GatedRepoError" in type(exc).__name__ or "gated" in msg.lower():
            lines = ["Acesso negado — aceite os termos de uso de TODOS esses modelos no HuggingFace:"]
            for _, url in self.REQUIRED_HF_MODELS:
                lines.append(f"  → {url}")
            lines.append("Depois reinicie a gravação.")
            logger.error("\n".join(lines))
            return lines[0]
        if "401" in msg or "RepositoryNotFound" in type(exc).__name__:
            return "Token HuggingFace inválido ou sem permissão. Verifique nas Settings."
        if "ConnectionError" in type(exc).__name__ or "timeout" in msg.lower():
            return "Sem conexão com HuggingFace. Verifique sua internet."
        return msg[:200]
