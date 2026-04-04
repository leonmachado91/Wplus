from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass
class WordTimestamp:
    word: str = ""
    start: float = 0.0
    end: float = 0.0


@dataclass
class TranscriptSegment:
    id: str = field(default_factory=lambda: f"seg-{uuid4().hex[:8]}")
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""
    speaker: Optional[str] = None
    confidence: float = 0.0
    words: list[WordTimestamp] = field(default_factory=list)
    chunk_was_forced: bool = False
    is_partial: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "speaker": self.speaker,
            "confidence": self.confidence,
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in self.words],
            "chunk_was_forced": self.chunk_was_forced,
        }
