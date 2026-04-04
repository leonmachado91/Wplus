"""Groq transcription engine — async client with rate limiting and retry."""

from __future__ import annotations

import asyncio
import io
import logging
import time
import threading
from typing import TYPE_CHECKING, Optional

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
        settings: SettingsManager,
        buffer: TranscriptBuffer,
        sample_rate: int = 16000,
    ) -> None:
        self._settings = settings
        self._buffer = buffer
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

        model = self._settings.get("api", "groq_model")
        language = self._settings.get("api", "groq_language")
        prompt = self._settings.get("api", "groq_prompt")
        temperature = self._settings.get("api", "groq_temperature")

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
                # Only pass language/prompt when explicitly set — empty string
                # causes Groq to reject the request with a 400 error.
                if language:
                    kwargs["language"] = language
                if prompt:
                    kwargs["prompt"] = prompt

                response = await self._client.audio.transcriptions.create(**kwargs)

                segment = self._response_to_segment(response, meta)
                if segment:
                    self._buffer.add_segment(segment)
                    logger.info("Transcribed: [%.1f-%.1f] %s", segment.start_time, segment.end_time, segment.text[:80])

                return

            except RateLimitError as e:
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

    def _response_to_segment(self, response: object, meta: dict) -> Optional[TranscriptSegment]:
        """Convert Groq verbose_json response to TranscriptSegment."""
        text = getattr(response, "text", "").strip()
        if not text:
            logger.debug("Empty transcription, skipping")
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
        seg_start = session_offset
        seg_end = meta.get("end_time", session_offset)
        avg_logprob = 0.0

        if raw_segments:
            first_seg = raw_segments[0]
            last_seg = raw_segments[-1]
            seg_start = session_offset + getattr(first_seg, "start", 0.0)
            seg_end = session_offset + getattr(last_seg, "end", 0.0)
            # average confidence from log probabilities
            logprobs = [getattr(s, "avg_logprob", 0.0) for s in raw_segments]
            if logprobs:
                import math
                avg_logprob = sum(logprobs) / len(logprobs)

        # convert avg_logprob to confidence (0-1)
        import math
        confidence = math.exp(avg_logprob) if avg_logprob else 0.0
        confidence = min(max(confidence, 0.0), 1.0)

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
