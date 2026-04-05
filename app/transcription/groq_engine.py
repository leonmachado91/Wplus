"""Groq transcription engine — async client with rate limiting and retry."""

from __future__ import annotations

import asyncio
import io
import logging
import time
import threading
from typing import TYPE_CHECKING, Callable, Optional

from groq import AsyncGroq, RateLimitError, APIStatusError

from app.transcription.segment import TranscriptSegment, WordTimestamp

if TYPE_CHECKING:
    from app.core.settings_manager import SettingsManager
    from app.core.transcript_buffer import TranscriptBuffer

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter for Groq free tier."""

    def __init__(self, max_requests: int = 20, window_seconds: float = 60.0) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            # purge old timestamps
            self._timestamps = [t for t in self._timestamps if now - t < self._window]

            if len(self._timestamps) >= self._max:
                oldest = self._timestamps[0]
                wait = self._window - (now - oldest) + 0.1
                logger.info("Rate limit: waiting %.1fs", wait)
                await asyncio.sleep(wait)

            self._timestamps.append(time.monotonic())


class TranscriptionEngine:
    """Consumes WAV chunks from a queue and transcribes via Groq API.

    Runs its own asyncio event loop in a daemon thread.
    """

    def __init__(
        self,
        settings: "SettingsManager",
        buffer: Optional["TranscriptBuffer"] = None,
        sample_rate: int = 16000,
        on_segment: Optional[Callable[["TranscriptSegment"], None]] = None,
    ) -> None:
        self._settings = settings
        self._buffer = buffer
        self._on_segment = on_segment
        self.sample_rate = sample_rate

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._async_queue: Optional[asyncio.Queue] = None
        self._rate_limiter = RateLimiter(max_requests=20, window_seconds=60.0)
        # _stop_event is created inside the asyncio loop (_process_loop).
        # We keep a threading.Event as a cross-thread signal so stop() works
        # even if called before the loop's asyncio.Event is ready.
        self._stop_requested = threading.Event()
        self._stop_event: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._client: Optional[AsyncGroq] = None

        # Rolling context: texto do último segmento válido (P1/P3)
        self._last_segment_text: str = ""
        self._last_segment_lock = threading.Lock()

    def reset_context(self) -> None:
        """P2: Limpa o contexto dinâmico entre sessões.

        Deve ser chamado sempre que uma nova sessão de gravação começa,
        para evitar que o contexto da sessão anterior influencie a nova.
        """
        with self._last_segment_lock:
            self._last_segment_text = ""
        logger.debug("Transcription context reset para nova sessão")

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="transcription-engine")
        self._thread.start()
        self._ready.wait(timeout=5)
        logger.info("Transcription engine started")

    def stop(self) -> None:
        self._stop_requested.set()
        if self._loop and self._stop_event:
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except Exception as e:
                logger.error("Error signalling groq stop event: %s", e)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Transcription engine stopped")

    def submit(self, wav_bytes: bytes, meta: dict) -> None:
        """Submit a WAV chunk for transcription (thread-safe)."""
        if self._loop and self._async_queue:
            asyncio.run_coroutine_threadsafe(
                self._async_queue.put((wav_bytes, meta)),
                self._loop,
            )

    # ── asyncio loop ─────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # We run the process loop as a task so we can cancel it cleanly
        self._task = self._loop.create_task(self._process_loop())
        try:
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    async def _process_loop(self) -> None:
        self._async_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()

        # single client for entire session — avoids event-loop-closed on cleanup
        api_key = self._settings.get("api", "groq_api_key")
        self._client = AsyncGroq(api_key=api_key) if api_key else None

        self._ready.set()

        try:
            while not self._stop_event.is_set() and not self._stop_requested.is_set():
                try:
                    wav_bytes, meta = await asyncio.wait_for(
                        self._async_queue.get(),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue

                await self._transcribe(wav_bytes, meta)

            # drain remaining items
            while not self._async_queue.empty():
                try:
                    wav_bytes, meta = self._async_queue.get_nowait()
                    await self._transcribe(wav_bytes, meta)
                except asyncio.QueueEmpty:
                    break
        finally:
            # close client while loop is still running
            if self._client:
                await self._client.close()
                self._client = None

    async def _transcribe(self, wav_bytes: bytes, meta: dict) -> None:
        if not self._client:
            logger.error("No Groq client — skipping transcription")
            return

        model       = self._settings.get("api", "groq_model")
        language    = self._settings.get("api", "groq_language")
        base_prompt = self._settings.get("api", "groq_prompt") or ""
        temperature = self._settings.get("api", "groq_temperature")

        # Rolling context: combina o prompt estático com o contexto dinâmico do último segmento.
        # O Whisper usa apenas os últimos ~224 tokens do prompt — truncamos para ≈1200 chars (~220 tok)
        # para garantir que a parte mais recente da conversa sempre caiba.
        with self._last_segment_lock:
            last_text = self._last_segment_text

        if last_text:
            dynamic_prompt = f"{base_prompt} {last_text}".strip() if base_prompt else last_text
        else:
            dynamic_prompt = base_prompt

        # Trunca pela direita para respeitar o limite de 224 tokens (~1200 chars)
        if len(dynamic_prompt) > 1200:
            dynamic_prompt = dynamic_prompt[-1200:]

        # retry with exponential backoff
        for attempt in range(3):
            try:
                await self._rate_limiter.acquire()

                file_obj = ("chunk.wav", io.BytesIO(wav_bytes), "audio/wav")

                kwargs: dict = dict(
                    model=model,
                    file=file_obj,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    temperature=temperature,
                )
                if language:
                    kwargs["language"] = language
                if dynamic_prompt:
                    kwargs["prompt"] = dynamic_prompt

                response = await self._client.audio.transcriptions.create(**kwargs)

                segment = self._response_to_segment(response, meta)
                if segment:
                    # Atualiza o contexto dinâmico APENAS se o segmento tem conteúdo suficiente.
                    # Chunks muito curtos (interjeições, ruídos) não devem contaminar o contexto
                    # porque erros de transcrição nesses chunks criam feedback loops.
                    segment_words = len(segment.text.split())
                    if segment_words >= 5:
                        with self._last_segment_lock:
                            combined = (self._last_segment_text + " " + segment.text).strip()
                            self._last_segment_text = combined[-600:] if len(combined) > 600 else combined

                    if self._buffer is not None:
                        self._buffer.add_segment(segment)
                    if self._on_segment is not None:
                        try:
                            self._on_segment(segment)
                        except Exception:
                            logger.exception("on_segment callback error")

                    conf_str = f"{segment.confidence:.2f}" if segment.confidence is not None else "n/a"
                    logger.info(
                        "Transcribed [%.1f-%.1fs] conf=%s words=%d: %s",
                        segment.start_time, segment.end_time, conf_str,
                        len(segment.text.split()), segment.text[:100]
                    )

                return

            except RateLimitError:
                wait = (2 ** attempt) * 2
                logger.warning("Groq rate limit hit, waiting %ds (attempt %d/3)", wait, attempt + 1)
                await asyncio.sleep(wait)

            except APIStatusError as e:
                if e.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning("Groq server error %d, retry in %ds", e.status_code, wait)
                    await asyncio.sleep(wait)
                else:
                    logger.error("Groq API error %d: %s", e.status_code, e.message)
                    return

            except Exception:
                logger.exception("Transcription error (attempt %d/3)", attempt + 1)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return

    def _is_hallucination(self, text: str, prev_text: str, confidence: float = 0.0) -> bool:
        """Detecta alucinações comuns do Whisper incluindo similaridade fuzzy.

        As listas de frases são lidas das settings em tempo de execução,
        permitindo que o usuário as configure pela UI sem reiniciar o app.
        """
        from difflib import SequenceMatcher

        text_lower = text.lower().strip()

        # Lê as listas das settings (configuráveis pela UI)
        filters = self._settings.get("filters")
        prefixes = [p.lower() for p in (filters.get("hallucination_prefixes") or [])]
        exact_set = {p.lower() for p in (filters.get("hallucination_exact") or [])}

        # 1. Prefixos: filtra quando a frase INICIA o texto
        for phrase in prefixes:
            if text_lower == phrase or text_lower.startswith(phrase + " "):
                return True

        # 2. Exact-match: só filtra quando é o texto COMPLETO do segmento
        # Ex: "e aí" sozinho = ruído; "e aí, você viu?" = fala real
        if text_lower in exact_set:
            return True

        # 3. Repetição exata do segmento anterior
        if prev_text and text_lower == prev_text.strip().lower():
            return True

        # 4. P3: Similaridade fuzzy com o segmento anterior (> 85%)
        if prev_text and len(text_lower) > 10 and len(prev_text) > 10:
            ratio = SequenceMatcher(None, text_lower, prev_text.lower().strip()).ratio()
            if ratio > 0.85:
                logger.debug("Similaridade %.0f%% com segmento anterior — descartado", ratio * 100)
                return True

        # 5. Texto muito curto (< 3 chars)
        if len(text_lower.replace(" ", "")) < 3:
            return True

        # 6. Palavra repetida 4+ vezes na mesma frase
        words = text_lower.split()
        if len(words) >= 4:
            for word in set(words):
                if words.count(word) >= 4 and len(word) > 2:
                    return True

        return False

    def _response_to_segment(self, response: object, meta: dict) -> Optional[TranscriptSegment]:
        """Convert Groq verbose_json response to TranscriptSegment."""
        text = getattr(response, "text", "").strip()
        if not text:
            logger.debug("Empty transcription, skipping")
            return None

        # Filtra alucinações antes de prosseguir (antes de calcular confidence)
        with self._last_segment_lock:
            prev = self._last_segment_text
        if self._is_hallucination(text, prev):
            logger.warning("Alucinação detectada e descartada: %r", text[:100])
            return None

        session_offset = meta.get("start_time", 0.0)
        chunk_was_forced = meta.get("chunk_was_forced", False)

        # extract word timestamps if available
        words: list[WordTimestamp] = []
        raw_words = getattr(response, "words", None) or []
        for w in raw_words:
            words.append(WordTimestamp(
                word=getattr(w, "word", ""),
                start=session_offset + getattr(w, "start", 0.0),
                end=session_offset + getattr(w, "end", 0.0),
            ))

        # extract segment-level timestamps
        raw_segments = getattr(response, "segments", None) or []

        # Fix: usa None como sentinel para avg_logprob
        # avg_logprob=0.0 é válido (significa log-prob perfeito), não pode ser usado como flag
        avg_logprob: "float | None" = None
        seg_start = session_offset

        # Fix: calcula end_time via duration_ms quando raw_segments não traz timestamps
        # (Groq frequentemente omite isso em chunks curtos, causando start==end no log)
        duration_s = meta.get("duration_ms", 0) / 1000.0
        seg_end = session_offset + duration_s if duration_s else session_offset

        if raw_segments:
            first_seg = raw_segments[0]
            last_seg = raw_segments[-1]
            raw_start = getattr(first_seg, "start", None)
            raw_end = getattr(last_seg, "end", None)
            if raw_start is not None:
                seg_start = session_offset + raw_start
            if raw_end is not None and raw_end > 0:
                seg_end = session_offset + raw_end

            logprobs = [getattr(s, "avg_logprob", None) for s in raw_segments]
            valid_logprobs = [lp for lp in logprobs if lp is not None]
            if valid_logprobs:
                avg_logprob = sum(valid_logprobs) / len(valid_logprobs)

        import math
        if avg_logprob is not None:
            confidence: "float | None" = min(max(math.exp(avg_logprob), 0.0), 1.0)
        else:
            confidence = None  # Groq não retornou logprob — não aplica filtro de threshold


        # P1: Filtro por confiança — só aplica se temos logprob real (não None)
        confidence_threshold = float(self._settings.get("api", "confidence_threshold") or 0.0)
        if confidence_threshold > 0.0 and confidence is not None and confidence < confidence_threshold:
            logger.warning(
                "Segmento descartado por baixa confiança: conf=%.3f < threshold=%.2f | %r",
                confidence, confidence_threshold, text[:80]
            )
            return None

        # Use the pre-assigned segment_id from meta so that the Diarization
        # engine (which received the same meta) can match and update this segment.
        seg_id = meta.get("segment_id")

        seg = TranscriptSegment(
            start_time=seg_start,
            end_time=seg_end,
            text=text,
            confidence=confidence,
            words=words,
            chunk_was_forced=chunk_was_forced,
            is_partial=False,
        )
        if seg_id:
            seg.id = seg_id
        return seg
