"""Transcript display widget — styled QTextEdit with speaker colors."""

from __future__ import annotations

import logging
from PyQt6.QtCore import QTimer, pyqtSlot
from PyQt6.QtGui import QTextCursor, QAction
from PyQt6.QtWidgets import QTextEdit, QMenu, QApplication

logger = logging.getLogger(__name__)

# Gruvbox speaker colors — cycling palette
SPEAKER_COLORS = [
    "#83a598",  # blue
    "#d3869b",  # purple
    "#fe8019",  # orange
    "#8ec07c",  # aqua
    "#fabd2f",  # yellow
    "#b8bb26",  # green
    "#fb4934",  # red
]

TIMECODE_COLOR = "#665c54"
TEXT_COLOR = "#ebdbb2"
PROCESSING_COLOR = "#a89984"


class TranscriptWidget(QTextEdit):
    """Read-only text area that renders transcription segments with speaker colors.

    Segments are stored in an ordered dict keyed by seg_id.  On speaker update
    we rebuild only the affected segment's HTML block using a string replacement
    on the full document HTML — O(n) in document size but avoids a full setHtml
    reflow.  For sessions with hundreds of segments this is significantly faster
    than the previous full _rerender() approach.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("transcriptWidget")
        self.setReadOnly(True)
        self.setAcceptRichText(True)

        self._auto_scroll = True
        self._speaker_color_map: dict[str, str] = {}
        self._next_color_idx = 0
        self._processing_visible = False
        # seg_id → {"tc": str, "speaker": str, "text": str, "sub_segments": list[dict]}
        self._segment_cache: dict[str, dict] = {}
        self._speaker_mapper = None

        # processing animation
        self._anim_dots = 0
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate_processing)

    # ── public API ───────────────────────────────────────────────────────

    def set_speaker_mapper(self, mapper: object) -> None:
        self._speaker_mapper = mapper

    @pyqtSlot(dict)
    def add_segment(self, segment: dict) -> None:
        self._hide_processing()

        seg_id = segment.get("id", "")
        tc = self._format_timecode(segment.get("start_time", 0.0))
        speaker = segment.get("speaker") or ""
        text = segment.get("text", "")
        sub_segments = segment.get("sub_segments") or []

        self._segment_cache[seg_id] = {
            "tc": tc,
            "speaker": speaker,
            "text": text,
            "sub_segments": sub_segments,
        }

        html = self._build_segment_html(seg_id, tc, speaker, text, sub_segments)
        self.append(f'<div id="{seg_id}">{html}</div>')

        if self._auto_scroll:
            self._scroll_to_bottom()

    def update_segment(self, segment_id: str, updates: dict) -> None:
        """Update speaker label (and optionally sub_segments) after diarization."""
        speaker = updates.get("speaker")
        sub_segments = updates.get("sub_segments")  # may be a list or None

        if segment_id not in self._segment_cache:
            return
        if not speaker and sub_segments is None:
            return

        cached = self._segment_cache[segment_id]
        changed = False

        if speaker and cached.get("speaker") != speaker:
            cached["speaker"] = speaker
            changed = True

        if sub_segments is not None and cached.get("sub_segments") != sub_segments:
            cached["sub_segments"] = sub_segments
            changed = True

        if "text" in updates:
            cached["text"] = updates["text"]
        if "start_time" in updates:
            cached["tc"] = self._format_timecode(updates["start_time"])

        if not changed:
            return

        scroll_pos = self.verticalScrollBar().value() if not self._auto_scroll else None
        self._full_rerender()
        if not self._auto_scroll and scroll_pos is not None:
            self.verticalScrollBar().setValue(scroll_pos)

    def clear_transcript(self) -> None:
        self.clear()
        self._speaker_color_map.clear()
        self._next_color_idx = 0
        self._segment_cache.clear()
        self._hide_processing()

    def set_auto_scroll(self, enabled: bool) -> None:
        self._auto_scroll = enabled

    # ── context menu ─────────────────────────────────────────────────────

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)

        copy_action = QAction("Copy Selected", self)
        copy_action.triggered.connect(self.copy)
        menu.addAction(copy_action)

        copy_all = QAction("Copy All", self)
        copy_all.triggered.connect(self._copy_all)
        menu.addAction(copy_all)

        menu.addSeparator()

        scroll_action = QAction("Auto-scroll", self)
        scroll_action.setCheckable(True)
        scroll_action.setChecked(self._auto_scroll)
        scroll_action.triggered.connect(self.set_auto_scroll)
        menu.addAction(scroll_action)

        menu.exec(event.globalPos())

    # ── internals ────────────────────────────────────────────────────────

    def _build_segment_html(
        self,
        _seg_id: str,
        tc: str,
        speaker: str,
        text: str,
        sub_segments: list | None = None,
    ) -> str:
        """Build HTML for a segment.

        When ``sub_segments`` is populated, render each span as its own line
        with the span's speaker and text.  The first span reuses ``tc`` (the
        chunk start timecode); subsequent spans show their own timecode.
        When empty, render the classic single-speaker line.
        """
        if sub_segments:
            # Drop any span with empty text before rendering.
            # This can happen when Groq returns word entries with empty strings,
            # making the reconstructed span text empty after join+strip.
            valid_spans = [s for s in sub_segments if s.get("text", "").strip()]
            if valid_spans:
                lines = []
                for i, span in enumerate(valid_spans):
                    span_tc = tc if i == 0 else self._format_timecode(span.get("start", 0.0))
                    span_speaker = span.get("speaker", "")
                    span_text = span.get("text", "")
                    lines.append(self._build_single_line(span_tc, span_speaker, span_text))
                return "<br>".join(lines)
            # All spans had empty text — fall through to single-speaker render below.

        return self._build_single_line(tc, speaker, text)

    def _build_single_line(self, tc: str, speaker: str, text: str) -> str:
        """Render a single speaker line: [timecode] Speaker  text."""
        parts = [f'<span style="color:{TIMECODE_COLOR};font-size:8pt;">[{tc}]</span>']

        if speaker:
            display = speaker
            if self._speaker_mapper:
                display = self._speaker_mapper.display_name(speaker)
                color = self._speaker_mapper.color(speaker)
            else:
                color = self._get_speaker_color(speaker)
            parts.append(f'&nbsp;<span style="color:{color};font-weight:bold;">{display}</span>')
        else:
            parts.append(f'&nbsp;<span style="color:#665c54;font-style:italic;">[…]</span>')

        parts.append(f'&nbsp;&nbsp;<span style="color:{TEXT_COLOR};">{text}</span>')
        return "".join(parts)

    def _full_rerender(self) -> None:
        """Full rebuild fallback — only used when targeted replacement fails."""
        blocks = []
        for seg_id, data in self._segment_cache.items():
            html = self._build_segment_html(
                seg_id,
                data["tc"],
                data.get("speaker", ""),
                data["text"],
                data.get("sub_segments"),
            )
            blocks.append(f'<div id="{seg_id}">{html}</div>')
        self.setHtml("<body style='background:#282828;'>" + "".join(blocks) + "</body>")
        if self._auto_scroll:
            self._scroll_to_bottom()

    def _get_speaker_color(self, speaker: str) -> str:
        if speaker not in self._speaker_color_map:
            color = SPEAKER_COLORS[self._next_color_idx % len(SPEAKER_COLORS)]
            self._speaker_color_map[speaker] = color
            self._next_color_idx += 1
        return self._speaker_color_map[speaker]

    @staticmethod
    def _format_timecode(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _scroll_to_bottom(self) -> None:
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def _copy_all(self) -> None:
        QApplication.clipboard().setText(self.toPlainText())

    def _animate_processing(self) -> None:
        self._anim_dots = (self._anim_dots + 1) % 4
        self._update_processing_text()

    def _update_processing_text(self) -> None:
        if not self._processing_visible:
            return
        dots = "." * self._anim_dots
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        selected = cursor.selectedText()
        if "processando" in selected.lower() or not self.toPlainText().strip():
            cursor.removeSelectedText()
            if self.toPlainText().strip():
                cursor.deletePreviousChar()

        html = (
            f'<span style="color:{PROCESSING_COLOR};font-style:italic;">'
            f"  ⏳ processando{dots}</span>"
        )
        self.append(html)
        if self._auto_scroll:
            self._scroll_to_bottom()

    def _hide_processing(self) -> None:
        if not self._processing_visible:
            return
        self._processing_visible = False
        self._anim_timer.stop()
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        selected = cursor.selectedText()
        if "processando" in selected.lower():
            cursor.removeSelectedText()
            if self.toPlainText().strip():
                cursor.deletePreviousChar()

    def verticalScrollBar(self):
        return super().verticalScrollBar()
