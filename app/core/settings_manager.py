from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "settings.json"


# ── Pydantic section models ─────────────────────────────────────────────────


class ApiSettings(BaseModel):
    groq_api_key: str = ""
    groq_model: str = "whisper-large-v3-turbo"
    groq_language: Optional[str] = None
    groq_prompt: str = ""
    groq_temperature: float = 0
    huggingface_token: str = ""


class AudioSettings(BaseModel):
    input_device_index: Optional[int] = None
    loopback_device_index: Optional[int] = None
    sample_rate: int = 16000
    channels: int = 1


class VADSettings(BaseModel):
    enabled: bool = True
    onset_threshold: float = 0.5
    offset_threshold: float = 0.35
    min_speech_duration_ms: int = 200
    max_chunk_duration_s: int = 15
    speech_pad_ms: int = 300


class DiarizationSettings(BaseModel):
    enabled: bool = False
    model: str = "pyannote/speaker-diarization-3.1"
    similarity_threshold: float = 0.65
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    speaker_labels: dict[str, str] = Field(default_factory=dict)


class ServerSettings(BaseModel):
    websocket_enabled: bool = True
    websocket_host: str = "127.0.0.1"
    websocket_port: int = 8765
    rest_api_enabled: bool = True
    rest_api_port: int = 8766
    api_key_enabled: bool = False
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "app://obsidian.md"]
    )


class Mode1Settings(BaseModel):
    watch_folder: str = ""
    output_folder: str = ""
    output_format: str = "md"
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"]
    )
    auto_start: bool = False


class Mode2Settings(BaseModel):
    audio_source: str = "mic"
    auto_save_enabled: bool = False
    auto_save_folder: str = ""
    show_timecodes: bool = True


class Mode3Settings(BaseModel):
    inject_method: str = "type"
    button_opacity: float = 0.85
    show_preview_bubble: bool = True


class UISettings(BaseModel):
    theme: str = "gruvbox_dark"
    font_family: str = "JetBrains Mono"
    font_size: int = 11
    minimize_to_tray: bool = True


class AppSettings(BaseModel):
    api: ApiSettings = Field(default_factory=ApiSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    diarization: DiarizationSettings = Field(default_factory=DiarizationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    mode1: Mode1Settings = Field(default_factory=Mode1Settings)
    mode2: Mode2Settings = Field(default_factory=Mode2Settings)
    mode3: Mode3Settings = Field(default_factory=Mode3Settings)
    ui: UISettings = Field(default_factory=UISettings)


# ── Settings Manager ────────────────────────────────────────────────────────


class SettingsManager:
    """Loads, validates and persists settings to a JSON file."""

    def __init__(self, path: Path | None = None):
        self._path = path or SETTINGS_PATH
        self._lock = threading.Lock()
        self._save_timer: Optional[threading.Timer] = None
        self.settings = self._load()

    # ── public API ───────────────────────────────────────────────────────

    def get(self, section: str, key: str | None = None) -> Any:
        with self._lock:
            section_obj = getattr(self.settings, section)
            if key is None:
                return section_obj.model_dump()
            return getattr(section_obj, key)

    def update(self, section: str, key: str, value: Any) -> None:
        with self._lock:
            section_obj = getattr(self.settings, section)
            setattr(section_obj, key, value)
        self._schedule_save()

    def update_section(self, section: str, data: dict) -> None:
        with self._lock:
            section_obj = getattr(self.settings, section)
            
            # Use Pydantic to cleanly coerce types (e.g. from UI QComboBox strings back to integers)
            current_dict = section_obj.model_dump()
            current_dict.update(data)
            
            model_class = type(section_obj)
            new_section_obj = model_class(**current_dict)
            setattr(self.settings, section, new_section_obj)
        self._schedule_save()

    def save_now(self) -> None:
        with self._lock:
            self._cancel_timer()
            self._write()

    def to_dict(self) -> dict:
        with self._lock:
            return self.settings.model_dump()

    def to_safe_dict(self) -> dict:
        """Returns settings dict with sensitive fields masked."""
        d = self.to_dict()
        if d["api"]["groq_api_key"]:
            d["api"]["groq_api_key"] = "***"
        if d["api"]["huggingface_token"]:
            d["api"]["huggingface_token"] = "***"
        return d

    # ── internal ─────────────────────────────────────────────────────────

    def _load(self) -> AppSettings:
        # load .env file if present
        env_path = self._path.parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                logger.info("Loaded .env from %s", env_path)
            except ImportError:
                # fallback: parse .env manually
                self._load_env_manual(env_path)

        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                settings = AppSettings.model_validate(raw)
                logger.info("Settings loaded from %s", self._path)
            except Exception:
                logger.exception("Failed to parse settings, using defaults")
                settings = AppSettings()
        else:
            settings = AppSettings()

        # env vars override settings.json
        import os
        env_groq = os.environ.get("GROQ_API_KEY", "")
        env_hf = os.environ.get("HUGGINGFACE_TOKEN", "")
        if env_groq and not settings.api.groq_api_key:
            settings.api.groq_api_key = env_groq
            logger.info("Groq API key loaded from environment")
        if env_hf and not settings.api.huggingface_token:
            settings.api.huggingface_token = env_hf
            logger.info("HuggingFace token loaded from environment")

        self._write(settings)
        return settings

    @staticmethod
    def _load_env_manual(path: Path) -> None:
        """Fallback .env parser when python-dotenv is not installed."""
        import os
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
        except Exception:
            logger.exception("Failed to parse .env manually")

    def _write(self, settings: AppSettings | None = None) -> None:
        s = settings or self.settings
        self._path.write_text(
            json.dumps(s.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Settings saved to %s", self._path)

    def _schedule_save(self) -> None:
        with self._lock:
            self._cancel_timer()
            self._save_timer = threading.Timer(0.5, self._debounced_save)
            self._save_timer.daemon = True
            self._save_timer.start()

    def _debounced_save(self) -> None:
        with self._lock:
            self._write()

    def _cancel_timer(self) -> None:
        if self._save_timer and self._save_timer.is_alive():
            self._save_timer.cancel()
            self._save_timer = None
