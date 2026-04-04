"""Converte e recorta arquivos de áudio grandes para os limites do Groq Whisper."""

from __future__ import annotations

import io
import logging
from typing import Iterator

import pydub

logger = logging.getLogger(__name__)

class AudioChunker:
    """Carrega arquivos estáticos via pydub, converte para 16kHz mono e devolve chunks em wav."""

    def __init__(self, target_sample_rate: int = 16000, chunk_duration_ms: int = 25000, overlap_ms: int = 2000):
        self.sample_rate = target_sample_rate
        self.chunk_size = chunk_duration_ms
        self.overlap = overlap_ms

    def slice_file(self, file_path: str) -> Iterator[tuple[bytes, float, float]]:
        """
        Carrega o arquivo e o recorta em pedaços consecutivos de `chunk_size`.
        Retorna iterador de (wav_bytes, duração_do_chunk_ms, offset_original_ms).
        """
        try:
            logger.info("Loading audio file: %s", file_path)
            audio = pydub.AudioSegment.from_file(file_path)
        except Exception as e:
            logger.error("Failed to load audio file: %s", e)
            return

        # convert to mono, target sr
        audio = audio.set_frame_rate(self.sample_rate).set_channels(1)

        total_ms = len(audio)
        if total_ms == 0:
            return

        logger.info("File loaded: %.2fs", total_ms / 1000.0)

        offset = 0
        while offset < total_ms:
            end = min(offset + self.chunk_size, total_ms)
            chunk = audio[offset:end]
            
            # export chunk to bytes
            buf = io.BytesIO()
            chunk.export(buf, format="wav", codec="pcm_s16le")
            wav_bytes = buf.getvalue()
            
            chunk_duration = len(chunk)
            yield wav_bytes, chunk_duration, offset

            # advance offset
            offset += (self.chunk_size - self.overlap)
