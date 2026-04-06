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
class SpeakerSpan:
    """Speaker attribution for a sub-range inside a TranscriptSegment.

    Populated when multiple speakers are detected within a single VAD chunk.
    ``start`` and ``end`` are in seconds from the session start (same reference
    as ``TranscriptSegment.start_time`` / ``end_time``).
    ``text`` is reconstructed from the word timestamps that fall in the span.
    """

    speaker: str = ""
    start: float = 0.0
    end: float = 0.0
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass
class TranscriptSegment:
    id: str = field(default_factory=lambda: f"seg-{uuid4().hex[:8]}")
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""
    speaker: Optional[str] = None
    confidence: Optional[float] = None  # None when Groq does not return avg_logprob
    words: list[WordTimestamp] = field(default_factory=list)
    chunk_was_forced: bool = False
    is_partial: bool = False
    # Populated only when >1 speaker is detected inside this chunk.
    # When empty, ``speaker`` holds the single speaker for the whole segment.
    sub_segments: list[SpeakerSpan] = field(default_factory=list)

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
            "sub_segments": [s.to_dict() for s in self.sub_segments],
        }
