from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from app.transcription.segment import TranscriptSegment

logger = logging.getLogger(__name__)

Listener = Callable[[str, dict], None]


class TranscriptBuffer:
    """Central store for the current session's transcript segments.

    All mutations go through this class so listeners (WebSocket, UI, file writer)
    are notified consistently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._segments: list[TranscriptSegment] = []
        self._listeners: list[Listener] = []
        self._session_id: Optional[str] = None
        self._pending_speakers: dict[str, str] = {}  # segment_id -> speaker_name

        # auto-save
        self._auto_save_path: Optional[Path] = None

    @property
    def session_id(self) -> Optional[str]:
        """Read-only access to the current session ID."""
        return self._session_id

    # ── listener management ──────────────────────────────────────────────

    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        self._listeners.remove(fn)

    def _notify(self, event: str, data: dict) -> None:
        for fn in self._listeners:
            try:
                fn(event, data)
            except Exception:
                logger.exception("Listener error on event %s", event)

    # ── session lifecycle ────────────────────────────────────────────────

    def start_session(self) -> str:
        with self._lock:
            self._session_id = uuid4().hex
            self._segments.clear()
            self._pending_speakers.clear()
            session_id = self._session_id
        self._notify("session_started", {"session_id": session_id})
        return session_id

    def stop_session(self) -> dict:
        with self._lock:
            info = {
                "session_id": self._session_id,
                "segment_count": len(self._segments),
                "duration_s": self._segments[-1].end_time if self._segments else 0.0,
            }
        self._notify("session_stopped", info)
        return info

    # ── segment operations ───────────────────────────────────────────────

    def add_segment(self, segment: TranscriptSegment) -> None:
        with self._lock:
            if segment.id in self._pending_speakers:
                segment.speaker = self._pending_speakers.pop(segment.id)
                logger.info("Applied pending speaker %s to segment %s", segment.speaker, segment.id)
            self._segments.append(segment)

        event = "segment_partial" if segment.is_partial else "segment_final"
        self._notify(event, {
            "session_id": self.session_id,
            "segment": segment.to_dict(),
        })
        # auto-save append — writes all lines for the segment (1 or N for sub_segments)
        if self._auto_save_path:
            try:
                lines = self._segment_to_lines(segment)
                with open(self._auto_save_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            except Exception:
                logger.exception("Auto-save failed")

    def update_segment(self, segment_id: str, **updates: object) -> Optional[TranscriptSegment]:
        with self._lock:
            seg = self._find(segment_id)
            if seg is None:
                if "speaker" in updates:
                    self._pending_speakers[segment_id] = str(updates["speaker"])
                    logger.debug("Segment %s not in buffer yet, storing speaker %s as pending", segment_id, updates["speaker"])
                return None
            for k, v in updates.items():
                setattr(seg, k, v)
            # Capture the dict and session_id inside the lock to prevent
            # a concurrent diarization write from producing an inconsistent snapshot.
            seg_dict = seg.to_dict()
            session_id = self._session_id

        self._notify("segment_updated", {
            "session_id": session_id,
            "segment": seg_dict,
        })

        # Auto-save: rewrite file when diarization updates arrive so the saved
        # file reflects the final speaker labels and sub_segments.
        if self._auto_save_path and ("speaker" in updates or "sub_segments" in updates):
            try:
                self._auto_save_rewrite()
            except Exception:
                logger.exception("Auto-save rewrite failed")

        return seg

    def get_segments(self) -> list[TranscriptSegment]:
        with self._lock:
            return list(self._segments)

    def get_plain_text(self, show_timecodes: bool = True, show_speakers: bool = True) -> str:
        lines: list[str] = []
        for seg in self.get_segments():
            lines.extend(
                self._segment_to_lines(seg, show_timecodes=show_timecodes, show_speakers=show_speakers)
            )
        return "\n".join(lines)

    def clear(self) -> None:
        with self._lock:
            self._segments.clear()
            self._pending_speakers.clear()

    # ── helpers ──────────────────────────────────────────────────────────

    def _find(self, segment_id: str) -> Optional[TranscriptSegment]:
        for seg in self._segments:
            if seg.id == segment_id:
                return seg
        return None

    def get_segment(self, segment_id: str) -> Optional[TranscriptSegment]:
        """Return a single segment by ID (thread-safe, read-only reference)."""
        with self._lock:
            return self._find(segment_id)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _segment_to_md_line(seg: "TranscriptSegment") -> str:
        """Legacy single-line format (used by auto-save header write)."""
        return TranscriptBuffer._segment_to_lines(seg)[0]

    @staticmethod
    def _segment_to_lines(
        seg: "TranscriptSegment",
        show_timecodes: bool = True,
        show_speakers: bool = True,
    ) -> list[str]:
        """Convert a segment to one or more markdown lines.

        When ``seg.sub_segments`` is populated, each speaker span becomes its
        own line with its own timecode.  Otherwise, a single line is returned
        (backwards-compatible with single-speaker segments).
        """
        def _tc(seconds: float) -> str:
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"[{h:02d}:{m:02d}:{s:02d}]"

        if seg.sub_segments:
            lines = []
            for i, span in enumerate(seg.sub_segments):
                parts: list[str] = []
                if show_timecodes:
                    # First span gets the chunk's start time; others get span start
                    t = seg.start_time if i == 0 else span.start
                    parts.append(_tc(t))
                if show_speakers and span.speaker:
                    parts.append(f"**{span.speaker}**")
                parts.append(span.text)
                lines.append("  ".join(parts))
            return lines

        # Single-speaker fallback
        parts = []
        if show_timecodes:
            parts.append(_tc(seg.start_time))
        if show_speakers and seg.speaker:
            parts.append(f"**{seg.speaker}**")
        parts.append(seg.text)
        return ["  ".join(parts)]

    # ── file export ──────────────────────────────────────────────────────

    def export_markdown(self, path: str | Path) -> Path:
        """Write all segments to a Markdown file, respecting sub_segments."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [f"# Transcription — Session {self._session_id or 'unknown'}\n"]
        for seg in self.get_segments():
            lines.extend(self._segment_to_lines(seg))
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Exported markdown to %s", p)
        return p

    def _auto_save_rewrite(self) -> None:
        """Rewrite the auto-save file with the current in-memory state.

        Called after diarization updates so the saved file always reflects
        the final speaker labels and sub_segments, not the draft written at
        add_segment time (before diarization arrives).
        """
        if not self._auto_save_path:
            return
        with self._lock:
            segs = list(self._segments)
            session_id = self._session_id
        header = f"# Transcription — Session {session_id or 'unknown'}\n\n"
        body_lines: list[str] = []
        for seg in segs:
            body_lines.extend(self._segment_to_lines(seg))
        try:
            self._auto_save_path.write_text(
                header + "\n".join(body_lines) + "\n",
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Auto-save rewrite write failed")

    def export_text(self, path: str | Path) -> Path:
        """Write all segments as plain text."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.get_plain_text(), encoding="utf-8")
        logger.info("Exported text to %s", p)
        return p

    def set_auto_save(self, path: str | Path | None) -> None:
        """Enable or disable auto-save. Pass None to disable."""
        if path is None:
            self._auto_save_path = None
            logger.info("Auto-save disabled")
        else:
            self._auto_save_path = Path(path)
            self._auto_save_path.parent.mkdir(parents=True, exist_ok=True)
            # write header
            with open(self._auto_save_path, "w", encoding="utf-8") as f:
                f.write(f"# Transcription — Session {self._session_id or 'unknown'}\n\n")
            logger.info("Auto-save enabled: %s", self._auto_save_path)
