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

        # Dedicated lightweight queue for RMS display in the UI.
        # The UI reads from this queue only — never from raw_pcm_queue —
        # so the VAD pipeline is never disrupted by display polling.
        # maxsize=50 is enough for ~2.5 seconds at 20 fps polling; older
        # values are dropped automatically when the queue is full.
        self.rms_queue: queue.Queue[float] = queue.Queue(maxsize=50)

        self._stream = None        # sd.InputStream for mic
        self._pyaudio = None       # pyaudiowpatch instance for loopback
        self._pyaudio_stream = None
        self._running = False
        self._is_loopback = False
        self._loopback_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._paused = False

        # Frame mixer — only active in 'both' mode.
        # Each source pushes to its own sub-queue; the mixer thread pairs frames
        # and adds them sample-by-sample before sending ONE frame to the VAD.
        self._mic_frames: Optional[queue.Queue] = None
        self._loop_frames: Optional[queue.Queue] = None
        self._mixer_thread: Optional[threading.Thread] = None
        self._aec_filter = None  # Software AEC engine

        # ── AGC (Automatic Gain Control) ────────────────────────────────────
        # Smoothed, per-session gain applied to mic frames only.  Noise-gated so
        # silence is never amplified.  disabled by default; enabled via start().
        self._agc_enabled: bool = False
        self._agc_target_rms: float = 0.12   # target level in float32 [-1,1] space
        self._agc_max_gain: float = 8.0      # absolute ceiling to prevent blowout
        self._agc_noise_gate: float = 0.003  # RMS below this = silence, skip
        self._agc_attack: float = 0.5        # fraction per frame — almost instant drop for loud peaks
        self._agc_release: float = 0.01      # fraction per frame — slow falloff/lift for quiet parts
        self._agc_gain: float = 1.0          # current smoothed gain (reset each session)

    # ── public API ───────────────────────────────────────────────────────

    def start(
        self,
        device_index: int | None = None,
        mode: str = "mic",
        use_windows_aec: bool = False,
        mic_normalize: bool = False,
    ) -> None:
        """Start capturing audio.

        Args:
            device_index: device index for the mic. None = system default.
            mode: 'mic' | 'loopback' | 'both'.
                  'both' starts mic AND WASAPI loopback simultaneously;
                  frames from both sources are interleaved in raw_pcm_queue
                  and the VAD processes them as a unified stream.
            use_windows_aec: when True, override ``device_index`` with the
                  Windows default *Communications* microphone, which causes
                  Windows to apply its AEC / NS / AGC pipeline before
                  delivering PCM. Falls back silently if the device cannot
                  be determined.
            mic_normalize: when True, apply software Automatic Gain Control
                  to mic frames.  The smoothed gain is reset each call so
                  back-to-back sessions start from unity gain.
        """
        # Reset AGC per-session — avoids carrying stale gain across recordings.
        self._agc_enabled = mic_normalize
        self._agc_gain = 1.0
        if mic_normalize:
            logger.info("Mic AGC enabled (target_rms=%.2f, max_gain=%.1fx)",
                        self._agc_target_rms, self._agc_max_gain)
        with self._lock:
            if self._running:
                logger.warning("Capture already running, ignoring start()")
                return

            if mode == "both":
                # Sub-queues for the frame mixer — must exist before streams start.
                self._mic_frames = queue.Queue(maxsize=50)
                self._loop_frames = queue.Queue(maxsize=50)
                
                if use_windows_aec:
                    from app.audio.software_aec import SoftwareAEC
                    self._aec_filter = SoftwareAEC()
                    logger.info("Software AEC (NLMS) filter initialized for 'both' mode")
                else:
                    self._aec_filter = None

                self._start_mic(device_index)
                try:
                    self._start_loopback()       # sets _pyaudio_stream + thread
                except Exception:
                    # Loopback failed after mic started — roll back everything.
                    logger.exception(
                        "Loopback failed in 'both' mode — stopping mic and re-raising"
                    )
                    if self._stream:
                        try:
                            self._stream.stop()
                            self._stream.close()
                        except Exception:
                            pass
                        self._stream = None
                    self._mic_frames = None
                    self._loop_frames = None
                    self._running = False
                    raise

                # Both streams running — start the frame mixer.
                self._mixer_thread = threading.Thread(
                    target=self._mixer_worker, daemon=True, name="frame-mixer"
                )
                self._mixer_thread.start()
                logger.info("Frame mixer started for 'both' mode")
            elif mode == "loopback":
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

                # In 'both' mode route to mixer; otherwise go direct to VAD.
                if self._loop_frames is not None:
                    try:
                        self._loop_frames.put_nowait(mono)
                    except queue.Full:
                        pass  # mixer fell behind — drop this loopback frame
                else:
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

            # Stop the frame mixer first (it drives the VAD feed).
            # The mixer thread checks _running and will exit within one frame (~50ms).
            if self._mixer_thread:
                self._mixer_thread.join(timeout=1)
                self._mixer_thread = None
            self._mic_frames = None
            self._loop_frames = None

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

        # Apply AGC before routing so both the mixer AND the VAD see normalized audio.
        frame = self._apply_agc(frame)

        # In 'both' mode, route to the frame mixer instead of directly to the VAD.
        if self._mic_frames is not None:
            try:
                self._mic_frames.put_nowait(frame)
            except queue.Full:
                pass  # mixer fell behind — drop this mic frame silently
        else:
            self._enqueue_frame(frame)

    # ── AGC ──────────────────────────────────────────────────────────

    def _apply_agc(self, frame: np.ndarray) -> np.ndarray:
        """Apply smoothed Automatic Gain Control to a single mic frame.

        Algorithm:
          1. Noise gate — frames below ``_agc_noise_gate`` RMS are silence;
             returning them unchanged avoids amplifying hiss between words.
          2. Desired gain = target_rms / frame_rms, capped at max_gain.
          3. Exponential smoothing with DIFFERENT attack/release rates:
             - attack is faster (gain goes up quickly when speech starts)
             - release is slower (gain falls gracefully, no sudden drop)
          4. Clip output to [−1, 1] to prevent clipping artifacts.

        Called from the sounddevice callback — must be lock-free and fast.
        Float arithmetic on a 512-sample frame takes ~3 µs on a modern CPU.
        """
        if not self._agc_enabled:
            return frame

        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms < self._agc_noise_gate:
            # Silence: hold current gain but don't amplify
            return frame

        desired_gain = min(self._agc_target_rms / (rms + 1e-9), self._agc_max_gain)

        # Asymmetric smoothing: attack fast (compress peaks), release slowly (recover gain)
        alpha = self._agc_attack if desired_gain < self._agc_gain else self._agc_release
        new_gain = self._agc_gain + alpha * (desired_gain - self._agc_gain)

        # Smooth interpolation per sample to avoid "zipper noise" at frame boundaries
        gain_curve = np.linspace(self._agc_gain, new_gain, len(frame), dtype=np.float32)
        if frame.ndim > 1:
            gain_curve = gain_curve.reshape(-1, 1)
            
        self._agc_gain = new_gain

        return np.clip(frame * gain_curve, -1.0, 1.0).astype(np.float32)

    def set_paused(self, paused: bool) -> None:
        """Pause or resume audio frame processing.

        When paused, frames are dropped silently before reaching the VAD queue,
        effectively halting transcription without tearing down the audio streams.
        """
        self._paused = paused
        logger.info("Capture paused: %s", self._paused)

    # ── shared queue write ───────────────────────────────────────────────

    def _enqueue_frame(self, frame: np.ndarray) -> None:
        if self._paused:
            return

        # Publish RMS for the UI display queue first (non-blocking, drops when full).
        try:
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            self.rms_queue.put_nowait(rms)
        except queue.Full:
            pass  # UI didn't read fast enough — silently drop the stale value

        # Then push the raw frame to the VAD pipeline queue.
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

    # ── frame mixer (both mode) ────────────────────────────────────

    def _mixer_worker(self) -> None:
        """Pair mic + loopback frames and mix by sample-wise addition.

        Root cause of the 'horror-movie glitch':
            When both sources push frames independently to the same VAD queue,
            the queue alternates between two completely different acoustic
            environments every 32 ms.  The speech buffer the VAD accumulates
            therefore oscillates between  room-mic silence and speaker audio —
            producing severe, unintelligible distortion.

        Fix:
            Drive on mic frames (sounddevice delivers at a steady ~32 ms cadence).
            For each mic frame, wait up to 20 ms for a matching loopback frame.
            If both arrive: add sample-by-sample and clip to [−1, 1].
            If only one arrives: push it alone (avoids stalling the VAD).
            Result: the VAD always receives a single temporally-coherent stream.
        """
        while self._running:
            # Drive on mic frames; sounddevice guarantees a steady cadence.
            try:
                mic_frame = self._mic_frames.get(timeout=0.05)   # type: ignore[union-attr]
            except (queue.Empty, AttributeError):
                continue  # engine stopping or queues cleared by stop()

            # Wait briefly for a matching loopback frame (within half a frame = 16 ms).
            try:
                loop_frame = self._loop_frames.get(timeout=0.016)  # type: ignore[union-attr]
            except (queue.Empty, AttributeError):
                loop_frame = None

            if loop_frame is not None:
                # Se houver AEC ativado, subtrai o som da caixa (loopback) do som do mic.
                if self._aec_filter is not None:
                    mic_clean = self._aec_filter.process_frame(mic_frame, loop_frame)
                else:
                    mic_clean = mic_frame
                    
                # Proper acoustic mixing: sample-by-sample addition, clip to valid range.
                mixed = np.clip(
                    mic_clean.astype(np.float64) + loop_frame.astype(np.float64),
                    -1.0, 1.0,
                ).astype(np.float32)
                self._enqueue_frame(mixed)
            else:
                # Loopback not yet ready (startup lag or clock drift) — use mic alone.
                self._enqueue_frame(mic_frame)
