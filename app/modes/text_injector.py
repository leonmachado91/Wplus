"""Text injector — types transcribed text into the currently focused window."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TextInjector:
    """Injects plain text into whatever text field currently has OS focus.

    Uses pynput.keyboard.Controller.type(), which sends individual key events
    via SendInput on Windows. This works on any native text field without
    stealing focus from the target window.

    Only the bare transcribed text is injected — no timestamps, no speaker labels.
    """

    def __init__(self, append_newline: bool = False) -> None:
        self._append_newline = append_newline
        self._controller: Optional[object] = None
        self._first_injection = True  # no leading space on the very first inject
        self._init_controller()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def _init_controller(self) -> None:
        try:
            from pynput.keyboard import Controller
            self._controller = Controller()
            logger.info("TextInjector: pynput controller ready")
        except Exception as e:
            logger.error("TextInjector: failed to init pynput — %s", e)
            self._controller = None

    # ── public API ────────────────────────────────────────────────────────

    @property
    def append_newline(self) -> bool:
        return self._append_newline

    @append_newline.setter
    def append_newline(self, value: bool) -> None:
        self._append_newline = value

    def inject(self, text: str) -> bool:
        """Type *text* into the currently focused OS text field.

        Prepends a single space so that consecutive injections don't merge
        words together (e.g. "hello" then "world" → "hello world").

        Args:
            text: Plain transcribed text. Must be non-empty after stripping.

        Returns:
            True if injection succeeded, False otherwise.
        """
        text = text.strip()
        if not text:
            return False

        if self._controller is None:
            logger.error("TextInjector: no controller available")
            return False

        try:
            # First injection: no leading space (cursor is already in the field).
            # Subsequent ones: prefix a space to separate utterances.
            if self._first_injection:
                payload = text
                self._first_injection = False
            else:
                payload = f" {text}"

            if self._append_newline:
                payload += "\n"

            self._controller.type(payload)  # type: ignore[attr-defined]
            logger.debug("TextInjector: injected %d chars", len(payload))
            return True

        except Exception as e:
            logger.error("TextInjector: injection failed — %s", e)
            return False
