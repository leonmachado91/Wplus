"""Audio capture — mic via sounddevice, WASAPI loopback via PyAudioWPatch."""

from __future__ import annotations

import logging
import queue
import struct
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

FRAME_SIZE = 512   # ~32 ms at 16 kHz
QUEUE_MAX = 500    # ~16 s buffer


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    is_loopback: bool
    default_samplerate: float


class AudioCaptureEngine:
    """Captures audio from mic or WASAPI loopback into a queue of float32 frames."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.raw_pcm_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=QUEUE_MAX)

        self._stream = None        # sd.InputStream for mic
        self._pyaudio = None       # pyaudiowpatch instance for loopback
        self._pyaudio_stream = None
        self._running = False
        self._is_loopback = False
        self._loopback_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── public API ───────────────────────────────────────────────────────

    def start(self, device_index: int | None = None, mode: str = "mic") -> None:
        """Start capturing audio.

        Args:
            device_index: device index. None = system default.
            mode: 'mic' for microphone, 'loopback' for system audio (WASAPI).
        """
        with self._lock:
            if self._running:
                logger.warning("Capture already running, ignoring start()")
                return

            if mode == "loopback":
                self._start_loopback()
            else:
                self._start_mic(device_index)

    def _start_mic(self, device_index: int | None) -> None:
        """Start capture from a microphone using sounddevice."""
        try:
            self._is_loopback = False
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=FRAME_SIZE,
                device=device_index,
                channels=self.channels,
                dtype="float32",
                callback=self._sd_callback,
            )
            self._stream.start()
            self._running = True
            logger.info(
                "Capture started: device=%s mode=mic sr=%d",
                device_index or "default",
                self.sample_rate,
            )
        except Exception:
            logger.exception("Failed to start mic capture")
            raise

    def _start_loopback(self) -> None:
        """Start WASAPI loopback capture using PyAudioWPatch."""
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise RuntimeError(
                "PyAudioWPatch is required for system audio capture. "
                "Install it with: pip install PyAudioWPatch"
            )

        try:
            self._pyaudio = pyaudio.PyAudio()
            wasapi_info = self._pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)

            # find default loopback device
            default_speakers = self._pyaudio.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )

            # search for the loopback counterpart
            loopback_device = None
            for i in range(self._pyaudio.get_device_count()):
                dev = self._pyaudio.get_device_info_by_index(i)
                if (
                    dev.get("isLoopbackDevice", False)
                    and dev["name"].startswith(default_speakers["name"].split(" (")[0])
                ):
                    loopback_device = dev
                    break

            if loopback_device is None:
                # fallback: pick any loopback device
                for i in range(self._pyaudio.get_device_count()):
                    dev = self._pyaudio.get_device_info_by_index(i)
                    if dev.get("isLoopbackDevice", False):
                        loopback_device = dev
                        break

            if loopback_device is None:
                raise RuntimeError("No WASAPI loopback device found")

            self._is_loopback = True
            self._loopback_sr = int(loopback_device["defaultSampleRate"])
            self._loopback_ch = loopback_device["maxInputChannels"]

            logger.info(
                "Loopback device: [%d] %s (%dch, %dHz)",
                loopback_device["index"],
                loopback_device["name"],
                self._loopback_ch,
                self._loopback_sr,
            )

            # Ensure we read enough native samples so that after downsampling we get exactly FRAME_SIZE samples
            # Example: to get 512 samples at 16kHz, from a 48kHz source, we need 512 * 48000 / 16000 = 1536 samples
            import math
            self._native_frames_per_buffer = math.ceil(FRAME_SIZE * (self._loopback_sr / self.sample_rate))

            self._pyaudio_stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self._loopback_ch,
                rate=self._loopback_sr,
                input=True,
                input_device_index=loopback_device["index"],
                frames_per_buffer=self._native_frames_per_buffer,
            )

            self._running = True

            # read in a separate thread to avoid blocking
            self._loopback_thread = threading.Thread(
                target=self._loopback_reader, daemon=True, name="loopback-reader"
            )
            self._loopback_thread.start()

            logger.info("Capture started: mode=loopback sr=%d", self._loopback_sr)

        except Exception:
            logger.exception("Failed to start loopback capture")
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None
            raise

    def _loopback_reader(self) -> None:
        """Thread that reads from PyAudioWPatch loopback stream."""
        from scipy.signal import resample_poly
        from math import gcd

        src_sr = self._loopback_sr
        dst_sr = self.sample_rate
        g = gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g

        while self._running and self._pyaudio_stream:
            try:
                raw = self._pyaudio_stream.read(self._native_frames_per_buffer, exception_on_overflow=False)
                # parse float32 samples
                n_samples = len(raw) // 4
                samples = np.array(
                    struct.unpack(f"{n_samples}f", raw), dtype=np.float32
                )

                # reshape to (frames, channels) and downmix to mono
                if self._loopback_ch > 1:
                    samples = samples.reshape(-1, self._loopback_ch)
                    mono = samples.mean(axis=1).astype(np.float32)
                else:
                    mono = samples

                # resample if native rate differs from target
                if src_sr != dst_sr:
                    mono = resample_poly(mono, up, down).astype(np.float32)

                # enqueue
                self._enqueue_frame(mono)

            except Exception:
                if self._running:
                    logger.debug("Loopback read error", exc_info=True)
                break

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False

            # stop sounddevice stream
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            # stop pyaudiowpatch loopback
            if self._pyaudio_stream:
                self._pyaudio_stream.stop_stream()
                
            if self._loopback_thread:
                self._loopback_thread.join(timeout=2)
                self._loopback_thread = None

            if self._pyaudio_stream:
                self._pyaudio_stream.close()
                self._pyaudio_stream = None
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None

            logger.info("Capture stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── device enumeration ───────────────────────────────────────────────

    @staticmethod
    def list_devices() -> list[DeviceInfo]:
        """List all available audio input devices (mic only, from sounddevice)."""
        devices: list[DeviceInfo] = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                devices.append(DeviceInfo(
                    index=i,
                    name=d["name"],
                    max_input_channels=d["max_input_channels"],
                    is_loopback=False,
                    default_samplerate=d["default_samplerate"],
                ))
        return devices

    @staticmethod
    def list_microphones() -> list[DeviceInfo]:
        return AudioCaptureEngine.list_devices()

    @staticmethod
    def list_loopback_devices() -> list[DeviceInfo]:
        """List WASAPI loopback devices via PyAudioWPatch."""
        devices: list[DeviceInfo] = []
        try:
            import pyaudiowpatch as pyaudio
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                d = p.get_device_info_by_index(i)
                if d.get("isLoopbackDevice", False):
                    devices.append(DeviceInfo(
                        index=d["index"],
                        name=d["name"],
                        max_input_channels=d["maxInputChannels"],
                        is_loopback=True,
                        default_samplerate=d["defaultSampleRate"],
                    ))
            p.terminate()
        except ImportError:
            logger.warning("PyAudioWPatch not installed — no loopback devices")
        except Exception:
            logger.exception("Error listing loopback devices")
        return devices

    # ── sounddevice callback (mic) ───────────────────────────────────────

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)

        # downmix to 1-D mono
        if indata.ndim > 1 and indata.shape[1] > 1:
            frame = indata.mean(axis=1).astype(np.float32)
        elif indata.ndim > 1:
            frame = indata[:, 0].copy()
        else:
            frame = indata.flatten().copy()

        self._enqueue_frame(frame)

    # ── shared queue write ───────────────────────────────────────────────

    def _enqueue_frame(self, frame: np.ndarray) -> None:
        try:
            self.raw_pcm_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.raw_pcm_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.raw_pcm_queue.put_nowait(frame)
            except queue.Full:
                pass
            logger.debug("Audio queue full — dropped oldest frame")
