"""Audio utility functions — PCM conversion, RMS, resampling."""

from __future__ import annotations

import io
import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert a float32 numpy array to WAV bytes (int16).

    Args:
        audio: 1-D float32 array with values in [-1, 1].
        sample_rate: Sample rate of the audio.

    Returns:
        bytes containing a valid WAV file.
    """
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return buf.read()


def get_rms(audio: np.ndarray) -> float:
    """Return the RMS level of an audio buffer (0.0 – 1.0 range)."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def resample_if_needed(
    audio: np.ndarray,
    from_sr: int,
    to_sr: int,
) -> np.ndarray:
    """Resample audio if sample rates differ, using scipy."""
    if from_sr == to_sr:
        return audio
    from scipy.signal import resample as scipy_resample

    num_samples = int(len(audio) * to_sr / from_sr)
    return scipy_resample(audio, num_samples).astype(np.float32)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return int(seconds * sample_rate)


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    return samples / sample_rate
