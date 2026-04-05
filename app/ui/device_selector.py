"""Device selector combo box with refresh button."""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QWidget

logger = logging.getLogger(__name__)


class DeviceSelector(QWidget):
    """Combo box for selecting audio input devices with a refresh button."""

    device_changed = pyqtSignal(object)  # emits device index (int or None)

    def __init__(self, mode: str = "all", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = mode  # 'all', 'mic', 'loopback'
        self._devices: list = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._combo = QComboBox()
        self._combo.setMinimumWidth(250)
        self._combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self._combo, 1)

        self._refresh_btn = QPushButton("↺")
        self._refresh_btn.setFixedSize(28, 28)
        self._refresh_btn.setToolTip("Atualizar lista de dispositivos")
        self._refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(self._refresh_btn)

        self.refresh()

    def refresh(self) -> None:
        """Re-enumerate audio devices and populate the combo box."""
        try:
            from app.audio.capture_engine import AudioCaptureEngine

            if self._mode == "mic":
                self._devices = AudioCaptureEngine.list_microphones()
            elif self._mode == "loopback":
                self._devices = AudioCaptureEngine.list_loopback_devices()
            else:
                self._devices = AudioCaptureEngine.list_devices()
        except Exception:
            logger.exception("Failed to enumerate devices")
            self._devices = []

        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("Padrão do Sistema", None)
        for d in self._devices:
            label = d.name
            if d.is_loopback:
                label += " [Loopback]"
            self._combo.addItem(label, d.index)
        self._combo.blockSignals(False)

    def selected_device_index(self) -> Optional[int]:
        return self._combo.currentData()

    def selected_device_name(self) -> str:
        return self._combo.currentText()

    def _on_selection_changed(self, index: int) -> None:
        self.device_changed.emit(self._combo.currentData())
