"""Diarization engine — per-chunk speaker verification via ECAPA-TDNN embeddings.

Abordagem: Speaker Verification, não Diarization.
    Cada chunk do VAD = uma utterance = um speaker.
    A pergunta é: "esse chunk é a mesma pessoa do chunk anterior?"

    ECAPA-TDNN (spkrec-ecapa-voxceleb) — EER 0.69% no VoxCeleb1.
    Top-K centroids (até 5 refs por speaker) para robustez em sessões longas.
    Watch mode: diarize_file() usa a pipeline AHC completa do pyannote.
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
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

DiarizationCallback = Callable[[str, list[dict]], None]  # (segment_id, annotations)

# Cosine similarity threshold for speaker re-identification.
# 0.70 balances within-speaker variability vs. cross-speaker separation for
# real-world mic/loopback audio (ECAPA VoxCeleb1 clean EER is at ~0.25, but
# real conditions shift the distribution — 0.35 is the practical sweet spot).
_SIM_THRESHOLD = 0.35

# Window size for embedding extraction.
# Long chunks are split into non-overlapping windows of this duration;
# any partial tail is discarded. Shorter chunks are embedded whole.
_WINDOW_S: float = 2.0

# Minimum RMS energy for a sub-window to be worth embedding.
# Skips silent/noisy windows before running the model.
_MIN_WINDOW_RMS: float = 0.005

# Maximum number of reference embeddings stored per speaker (top-K centroids).
# More robust than a running average for long sessions — avoids drift.
_MAX_SPEAKER_REFS: int = 5


class DiarizationEngine:
    """Per-chunk speaker verification using ECAPA-TDNN (SpeechBrain).

    Live mode: ECAPA-TDNN loaded directly — fast startup, no pyannote pipeline.
    Watch mode: diarize_file() uses full pyannote AHC pipeline for offline files.

    Lazy model loading in a daemon thread. Processes one chunk at a time
    from a queue — no UI blocking, no OOM risk.
    """

    def __init__(
        self,
        hf_token: str,
        on_result: DiarizationCallback,
        on_status: Callable[[str], None] | None = None,
        use_gpu: bool = False,
        similarity_threshold: float = _SIM_THRESHOLD,
    ):
        self._token = hf_token
        self._on_result = on_result
        self._on_status = on_status or (lambda msg: None)
        self._use_gpu = use_gpu
        self._similarity_threshold = similarity_threshold

        # Loaded lazily in the worker thread
        self._encoder = None          # SpeechBrain ECAPA-TDNN
        self._model_loaded = threading.Event()
        self._model_error: Optional[str] = None
        self._load_lock = threading.Lock()

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False

        # ── Per-session speaker registry ───────────────────────────────────
        # List of (list_of_ref_embeddings, global_speaker_id)
        # Each speaker stores up to _MAX_SPEAKER_REFS recent embeddings.
        self._known_speakers: list[tuple[list[np.ndarray], str]] = []
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
            # Não bloquear por muito tempo a UI
            self._thread.join(timeout=2)
        self._is_running = False
        logger.info("DiarizationEngine stopped.")

    def submit(self, wav_bytes: bytes, chunk_meta: dict, segment_id: str) -> None:
        """Submit a WAV chunk (non-blocking). Drops oldest if queue > 4."""
        if not self._is_running:
            return
        if self._queue.qsize() >= 4:
            logger.warning("Diarization queue full — dropping chunk %s", segment_id)
            return
    def reset_speakers(self) -> None:
        """Clear speaker history (call at session start/stop)."""
        with self._speaker_lock:
            self._known_speakers.clear()
            self._next_speaker_idx = 0
        logger.debug("DiarizationEngine: speaker registry reset.")

    # ── Módulo Standalone "Soft-matching" ────────────────────────────

    def identify_speakers_sync(self, audio_np: np.ndarray, sr: int = 16000) -> list[tuple[str, float]]:
        """Uso síncrono da Diarization. (Para ser chamado pelo worker do ModeController).
        
        Args:
            audio_np: Chunk da Track já isolada.
            sr: Taxa de amostragem.
            
        Returns:
            Lista de (Speaker_ID, Similaridade), ordenados do maior pro menor.
            O próprio `match_or_register` fará o split se não houver um bom match.
        """
        self._load_model()
        if not self._encoder:
            return []

        embedding = self._averaged_embedding(audio_np, sr)
        if embedding is None:
            return []

        return self._match_or_register(embedding)

    def has_overlapping_speakers(self, audio_np: np.ndarray, sr: int = 16000, threshold: float = 0.15) -> bool:
        """
        Overlap Detector Booleano ultrarrápido (O Porteiro).
        Fatia o áudio em janelas gigantes (1.5s) que possuem impressão local forte o suficiente 
        para diferenciar biometrias vocais.
        
        Se a similaridade cruzada mínima da janela com o restante mergulhar abaixo do threshold, 
        confirma que as impressões variaram estruturalmente: há Overlap de duas ou mais pessoas.
        """
        self._load_model()
        if not self._encoder:
            # Fallback tolerante a falhas (transcreve tudo misturado sem ativar separador)
            return False

        win_samples = int(1.5 * sr)
        step_samples = win_samples // 2 
        total_samples = len(audio_np)

        if total_samples < win_samples:
            return False # Áudio curtíssimo, assumir 1 pessoa.

        embs = []
        pos = 0
        while pos + win_samples <= total_samples:
            chunk = audio_np[pos : pos + win_samples]
            pos += step_samples
            
            # Limpa espasmos/fantasmas na fatia para não gerar vectores de ruído de sala
            # Aumentamos o gate para 0.02 (evita embular como "2a pessoa" ruídos de fundo ou respiração mútua)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < 0.02: 
                continue
                
            emb = self._extract_embedding(chunk, sr)
            if emb is not None:
                embs.append(emb)

        if len(embs) < 2:
            return False # Só 1 janela válida salva -> Falso (só 1 pessoa)

        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims.append(np.dot(embs[i], embs[j]))

        min_sim = float(np.min(sims))
        
        # O limite prova-se rigoroso suficiente para separar 2 pessoas (tipicamente < 0.05) 
        # sem dar falso positivo para oscilações vocais da mesma pessoa (tipicamente > 0.3)
        if min_sim < threshold:
            logger.debug(f"[Overlap Porteiro] OVERLAP IDENTIFICADO! Mínimo cruzado {min_sim:.3f} afundou do limiar de {threshold:.2f}.")
            return True
        else:
            logger.debug(f"[Overlap Porteiro] Liberado. Apenas 1 pessoa detectada. Mínimo cruzado {min_sim:.3f}")
            return False

    def diarize_file(self, audio_path: str) -> list[dict]:
        """Offline diarization using pyannote full pipeline — Watch mode only.

        Uses AHC clustering with the entire recording available — much superior
        to live mode since it has global context of all speakers in the file.

        Returns:
            List of dicts: [{"speaker": str, "start": float, "end": float}, ...]
        """
        import warnings

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError:
            raise RuntimeError(
                "pyannote.audio not installed. Run: pip install pyannote.audio>=3.3.2"
            )

        logger.info("Loading pyannote pipeline for file diarization: %s", audio_path)
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

        if self._use_gpu and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            logger.info("pyannote pipeline on CUDA.")

        diarization = pipeline(audio_path)
        elapsed = time.monotonic() - t0
        logger.info("File diarization completed in %.1fs", elapsed)

        return [
            {"speaker": turn_label, "start": turn.start, "end": turn.end}
            for turn, _, turn_label in diarization.itertracks(yield_label=True)
        ]

    # ── worker ────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        self._load_model()
        if not self._encoder:
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
        """Identifica o speaker do chunk inteiro via ECAPA-TDNN."""
        try:
            import soundfile as sf

            wav_io = io.BytesIO(wav_bytes)
            wav_io.seek(0)
            audio_np, sr = sf.read(wav_io, dtype="float32")

            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            duration_s = len(audio_np) / sr
            if duration_s < 0.5:
                logger.debug("Chunk %s curto demais (%.2fs) — ignorado", segment_id, duration_s)
                return

            session_start = chunk_meta.get("start_time", 0.0)
            t0 = time.monotonic()

            embedding = self._averaged_embedding(audio_np, sr)
            if embedding is None:
                logger.warning("Sem embedding válido para %s", segment_id)
                return

            speaker_id = self._match_or_register(embedding)
            annotations = [{
                "speaker": speaker_id,
                "start": session_start,
                "end": session_start + duration_s,
            }]

            elapsed = time.monotonic() - t0
            logger.info(
                "Diarization: %s → %s em %.2fs (%.2fs áudio)",
                segment_id, speaker_id, elapsed, duration_s,
            )
            self._on_result(segment_id, annotations)

        except Exception as e:
            logger.error("Erro ao processar chunk %s: %s", segment_id, e, exc_info=True)

    def _averaged_embedding(self, audio_np: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Return a single unit-norm embedding representing the whole chunk.

        For chunks longer than _WINDOW_S: split into non-overlapping windows of
        exactly _WINDOW_S (any partial tail is discarded), extract one embedding
        per window, and return the L2-normalised mean.

        For shorter chunks: embed the whole audio directly.

        Averaging over multiple windows produces a more stable speaker
        representation than any individual window, and eliminates the partial-
        tail problem that caused false new-speaker registrations.
        """
        window_samples = int(_WINDOW_S * sr)
        total_samples = len(audio_np)

        if total_samples < window_samples:
            # Short chunk — embed whole audio
            return self._extract_embedding(audio_np, sr)

        embeddings: list[np.ndarray] = []
        pos = 0
        while pos + window_samples <= total_samples:
            window_np = audio_np[pos : pos + window_samples]

            rms = float(np.sqrt(np.mean(window_np ** 2)))
            if rms < _MIN_WINDOW_RMS:
                logger.debug("Window at %.2fs skipped: RMS %.4f below threshold", pos / sr, rms)
                pos += window_samples
                continue

            emb = self._extract_embedding(window_np, sr)
            if emb is not None:
                embeddings.append(emb)

            pos += window_samples  # non-overlapping step

        if not embeddings:
            return None

        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 1e-8 else None

    def _extract_embedding(self, audio_np: np.ndarray, _sr: int = 16000) -> Optional[np.ndarray]:
        """Extract unit-norm speaker embedding using ECAPA-TDNN (SpeechBrain)."""
        import torch
        try:
            tensor = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, samples)
            # encode_batch returns (batch, 1, embedding_dim)
            emb = self._encoder.encode_batch(tensor).squeeze().detach().cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                return None
            return emb / norm  # unit vector for cosine similarity via dot product
        except Exception as exc:
            logger.warning("Embedding error: %s", exc)
            return None

    def _match_or_register(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """Compara embedding com speakers conhecidos e retorna o ranking 'Soft-Match'.
        
        Se a similaridade do melhor bater com o threshold, registra o embedding no histórico.
        Retorno: [(Speaker A, 0.82), (Speaker B, 0.65)]
        """
        with self._speaker_lock:
            if self._known_speakers:
                # Rankea todos as similaridades
                candidates = []
                for refs, speaker_id in self._known_speakers:
                    sim = float(np.mean([np.dot(embedding, ref) for ref in refs]))
                    candidates.append((speaker_id, sim))
                    
                # Sort por similaridade DESC
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_id, best_sim = candidates[0]

                if best_sim >= self._similarity_threshold:
                    # Registra pro Speaker e Retorna
                    for refs, speaker_id in self._known_speakers:
                        if speaker_id == best_id:
                            refs.append(embedding.copy())
                            if len(refs) > _MAX_SPEAKER_REFS:
                                refs.pop(0)
                            break
                    return candidates
                
                # Se não bateu com nenhum o suficiente, ele é um NOVO speaker,
                # mas ainda assim vamos passar adiante a lista inteira (o novo entra no top 1)

            # Novo speaker
            global_id = f"Speaker {self._next_speaker_idx}"
            self._next_speaker_idx += 1
            self._known_speakers.append(([embedding.copy()], global_id))
            logger.info("Novo speaker registrado: %s (%d conhecidos)", global_id, len(self._known_speakers))
            
            # Se havia conhecidos, concatena. Se não, devolve direto.
            # Um novo speaker tem confidence 1.0 (é ele mesmo).
            if self._known_speakers and len(self._known_speakers) > 1:
                 # Calcula o embedding dele contra os antigos para a "segunda" opção
                 candidates = [(global_id, 1.0)]
                 for refs, speaker_id in self._known_speakers[:-1]:
                     sim = float(np.mean([np.dot(embedding, ref) for ref in refs]))
                     candidates.append((speaker_id, sim))
                 candidates.sort(key=lambda x: x[1], reverse=True)
                 return candidates
            
            return [(global_id, 1.0)]

    # ── model loading ─────────────────────────────────────────────────────

    def _ensure_dependencies(self) -> bool:
        """Ensure speechbrain is available (installed as dep of pyannote.audio)."""
        try:
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]
        except ImportError:
            pass

        try:
            import speechbrain  # noqa: F401
            return True
        except ImportError:
            pass

        self._on_status("Instalando speechbrain (pode levar alguns minutos)…")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "speechbrain", "--quiet"],
                timeout=300,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self._model_error = f"Falha ao instalar speechbrain: {e}"
            logger.error(self._model_error)
            return False

    REQUIRED_HF_MODELS = [
        ("pyannote/speaker-diarization-3.1",      "https://huggingface.co/pyannote/speaker-diarization-3.1"),
        ("pyannote/segmentation-3.0",             "https://huggingface.co/pyannote/segmentation-3.0"),
    ]

    def _load_model(self) -> None:
        """Load ECAPA-TDNN directly via SpeechBrain for live speaker verification."""
        with self._load_lock:
            if self._encoder is not None:
                return

            if not self._ensure_dependencies():
                return

        import warnings

        try:
            import torch
            import huggingface_hub as _hfhub
            from huggingface_hub import snapshot_download
            from speechbrain.inference.speaker import EncoderClassifier
            import speechbrain.utils.fetching as _sb_fetching

            device = "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
            models_dir = Path(os.environ.get("TORCH_HOME", ".models/torch")).parent
            savedir = models_dir / "speechbrain" / "ecapa-voxceleb"
            savedir.mkdir(parents=True, exist_ok=True)

            if self._token and not os.environ.get("HF_TOKEN"):
                os.environ["HF_TOKEN"] = self._token

            self._on_status("Carregando modelo de diarização…")
            logger.info("Loading ECAPA-TDNN (SpeechBrain) on %s...", device.upper())
            t0 = time.monotonic()

            # SpeechBrain's fetcher calls hf_hub_download(use_auth_token=...) but
            # newer huggingface_hub removed that param in favour of token=.
            # Patch both the module-level ref and SpeechBrain's local import.
            _orig_dl = _hfhub.hf_hub_download
            def _compat_dl(*a, use_auth_token=None, **kw):  # noqa: E301
                if use_auth_token is not None and "token" not in kw:
                    kw["token"] = use_auth_token
                return _orig_dl(*a, **kw)
            _hfhub.hf_hub_download = _compat_dl
            _sb_fetching.hf_hub_download = _compat_dl

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*libtorchcodec.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*SYMLINK.*", category=UserWarning)

                try:
                    # snapshot_download caches all files that exist in the repo.
                    # After this call, hf_hub_download will find files in cache
                    # and skip network requests.
                    model_path = snapshot_download(
                        repo_id="speechbrain/spkrec-ecapa-voxceleb",
                        cache_dir=str(models_dir / "huggingface" / "hub"),
                        token=self._token or None,
                    )

                    # custom.py doesn't exist in this repo — stub so SpeechBrain's
                    # fetcher finds it locally and skips the HF Hub request.
                    _custom_py = Path(model_path) / "custom.py"
                    if not _custom_py.exists():
                        _custom_py.touch()

                    self._encoder = EncoderClassifier.from_hparams(
                        source=model_path,
                        savedir=str(savedir),
                        run_opts={"device": device},
                    )
                finally:
                    _hfhub.hf_hub_download = _orig_dl
                    _sb_fetching.hf_hub_download = _orig_dl

            logger.info("ECAPA-TDNN carregado em %.1fs no %s", time.monotonic() - t0, device.upper())

            elapsed = time.monotonic() - t0
            logger.info("Diarization engines prontos em %.1fs total", elapsed)
            self._model_loaded.set()
            self._on_status("Diarização pronta.")

        except Exception as e:
            self._encoder = None
            self._model_error = self._humanize_error(e)
            logger.error("Failed to load ECAPA-TDNN: %s", self._model_error)
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
