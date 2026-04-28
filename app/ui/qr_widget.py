"""QR code display widget for session pairing."""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class QRWidget(QWidget):
    """Displays a QR code PNG (bytes) inside a fixed-size label."""

    def __init__(self, size: int = 200, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._size = size
        self._label = QLabel()
        self._label.setFixedSize(size, size)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "background: #ffffff; border-radius: 8px; padding: 4px;"
        )
        self._url_label = QLabel()
        self._url_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._url_label.setStyleSheet("color: #a89984; font-size: 11px;")
        self._url_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._url_label)

        self.clear()

    def set_qr(self, png_bytes: bytes, url: str = "") -> None:
        """Render a QR code from raw PNG bytes."""
        pix = QPixmap()
        pix.loadFromData(png_bytes, "PNG")
        self._label.setPixmap(
            pix.scaled(
                self._size,
                self._size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._url_label.setText(url)

    def clear(self) -> None:
        self._label.clear()
        self._label.setText("QR aqui")
        self._label.setStyleSheet(
            "color: #7c6f64; background: #3c3836; border-radius: 8px; font-size: 12px;"
        )
        self._url_label.setText("")
