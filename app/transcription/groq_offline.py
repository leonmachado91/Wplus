"""Groq transcription wrapper for offline, bulk processing (Mode 1)."""

from __future__ import annotations

import io
import logging
from typing import Any
import time

from groq import Groq, RateLimitError, APIStatusError

from app.core.settings_manager import SettingsManager

logger = logging.getLogger(__name__)

class GroqOfflineTranscriber:
    """Synchronous Groq un-buffered transcriber for Mode 1."""
    
    def __init__(self, settings: SettingsManager):
        self._settings = settings
        api_key = self._settings.get("api", "groq_api_key")
        self.client = Groq(api_key=api_key) if api_key else None
        
    def transcribe_chunk(self, wav_bytes: bytes) -> str:
        if not self.client:
            logger.error("No Groq client configured. Cannot transcribe offline chunk.")
            return ""

        model = self._settings.get("api", "groq_model")
        language = self._settings.get("api", "groq_language")
        prompt = self._settings.get("api", "groq_prompt")
        temperature = self._settings.get("api", "groq_temperature")
        
        for attempt in range(3):
            try:
                # Groq has a rate limit per minute. We handle it with simple sleep.
                file_obj = ("chunk.wav", io.BytesIO(wav_bytes), "audio/wav")
                
                kwargs: dict[str, Any] = {
                    "file": file_obj,
                    "model": model,
                    "response_format": "text",
                    "temperature": temperature,
                }
                if language:
                    kwargs["language"] = language
                if prompt:
                    kwargs["prompt"] = prompt

                logger.info("Transcribing chunk...")
                t0 = time.monotonic()
                response = self.client.audio.transcriptions.create(**kwargs)
                logger.debug("Transcribed in %.2fs", time.monotonic() - t0)
                
                if isinstance(response, str):
                    return response.strip()
                elif hasattr(response, "text"):
                    return response.text.strip()
                return str(response).strip()
                
            except RateLimitError as e:
                wait_time = 3 * (2 ** attempt)
                logger.warning("Groq rate limit: waiting %ds...", wait_time)
                time.sleep(wait_time)
            except APIStatusError as e:
                logger.error("Groq API error: %s", e)
                break
            except Exception as e:
                logger.error("Unknown transcription error: %s", e)
                break

        return ""
