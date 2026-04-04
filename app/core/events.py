from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class AppEvent:
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass(frozen=True)
class SessionStartedEvent(AppEvent):
    session_id: str = ""
    mode: str = ""
    audio_source: str = ""


@dataclass(frozen=True)
class SessionStoppedEvent(AppEvent):
    session_id: str = ""
    segment_count: int = 0
    duration_s: float = 0.0


@dataclass(frozen=True)
class SegmentAddedEvent(AppEvent):
    session_id: str = ""
    segment_id: str = ""
    text: str = ""
    is_partial: bool = False


@dataclass(frozen=True)
class SegmentUpdatedEvent(AppEvent):
    session_id: str = ""
    segment_id: str = ""
    speaker: str = ""


@dataclass(frozen=True)
class StatusUpdateEvent(AppEvent):
    status: str = ""
    detail: str = ""


@dataclass(frozen=True)
class ErrorEvent(AppEvent):
    code: str = ""
    message: str = ""
    recoverable: bool = True
    detail: Any = None


@dataclass(frozen=True)
class FileTranscriptionStartedEvent(AppEvent):
    file_path: str = ""


@dataclass(frozen=True)
class FileTranscriptionCompleteEvent(AppEvent):
    file_path: str = ""
    output_path: str = ""
    segment_count: int = 0
