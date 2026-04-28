"""Streaming WebM/Opus decoder using a persistent ffmpeg subprocess per participant."""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Must match VADProcessor's expected frame size (512 samples @ 16 kHz ≈ 32 ms)
FRAME_SIZE = 512
SAMPLE_RATE = 16000


class OpusStreamDecoder:
    """Decodes MediaRecorder WebM/Opus blobs into PCM via a persistent ffmpeg pipe.

    MediaRecorder produces a valid WebM stream: the first ondataavailable event
    contains the EBML header + first cluster; subsequent events contain only
    clusters. Feeding all blobs sequentially to ffmpeg's stdin handles this
    correctly without any client-side header management.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._proc: Optional[subprocess.Popen] = None
        self._output_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg_exe = get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        self._stop_event.clear()
        self._proc = subprocess.Popen(
            [
                ffmpeg_exe,
                "-f", "webm", "-i", "pipe:0",
                "-f", "s16le",
                "-ar", str(self._sample_rate),
                "-ac", "1",
                "pipe:1",
                "-loglevel", "quiet",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f"opus-reader-{id(self)}",
        )
        self._reader_thread.start()
        logger.debug("OpusStreamDecoder started (ffmpeg pid %d)", self._proc.pid)

    def _read_loop(self) -> None:
        assert self._proc is not None
        # Read in chunks sized for one 512-sample frame (2 bytes per int16 sample)
        read_size = FRAME_SIZE * 2
        while not self._stop_event.is_set():
            try:
                data = self._proc.stdout.read(read_size)
                if not data:
                    break
                self._output_queue.put(data)
            except Exception:
                break

    def write(self, webm_bytes: bytes) -> None:
        """Write a WebM blob to ffmpeg stdin. Non-blocking from caller's view."""
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(webm_bytes)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            logger.warning("OpusStreamDecoder: ffmpeg pipe broken")

    def drain(self) -> np.ndarray:
        """Return all PCM samples currently available in the output queue (float32)."""
        pcm_bytes = bytearray()
        while True:
            try:
                data = self._output_queue.get_nowait()
                pcm_bytes.extend(data)
            except queue.Empty:
                break
        if pcm_bytes:
            return np.frombuffer(bytes(pcm_bytes), dtype=np.int16).astype(np.float32) / 32768.0
        return np.array([], dtype=np.float32)

    def close(self) -> None:
        self._stop_event.set()
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        logger.debug("OpusStreamDecoder closed")
