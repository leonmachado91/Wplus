"""Floating Button panel — tab content for Mode 3 (voice dictation)."""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from PyQt6.QtCore import Qt, QMetaObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.device_selector import DeviceSelector
from app.ui.floating_window import FloatingWindow

if TYPE_CHECKING:
    from app.core.settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class FloatingButtonPanel(QWidget):
    """Panel shown in the 'Floating Button' tab.

    Activating this tab minimizes the main window and shows the FloatingWindow.
    All recording happens through the FloatingWindow button — this panel shows
    session history and device selection only.
    """

    # Thread-safe signals
    _text_injected_sig = pyqtSignal(str)
    _reset_injector_sig = pyqtSignal()  # tells injection worker to reset first-space logic

    def __init__(
        self,
        settings: "SettingsManager",
        mode_controller: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._mode = mode_controller

        self._floating_win: Optional[FloatingWindow] = None
        self._is_recording = False
        self._injected_count = 0

        # Dedicated thread + queue for text injection.
        # Keeps pynput calls out of the asyncio thread (avoids event-loop blocking)
        # and out of the Qt main thread (avoids UI jank).
        self._inject_queue: queue.Queue[str] = queue.Queue()
        self._reset_event = threading.Event()  # signals worker to reset TextInjector
        self._inject_thread = threading.Thread(
            target=self._injection_worker,
            daemon=True,
            name="text-injector-thread",
        )
        self._inject_thread.start()

        self._build_ui()
        self._connect_signals()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # ── info label ───────────────────────────────────────────────────
        info = QLabel(
            "Clique na aba para ativar o botão flutuante.\n"
            "O botão fica sempre visível sobre outras janelas.\n"
            "Clique no botão → fale → o texto é digitado onde estiver o cursor."
        )
        info.setObjectName("labelMuted")
        info.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── device row ───────────────────────────────────────────────────
        device_row = QHBoxLayout()
        device_label = QLabel("Dispositivo:")
        device_label.setObjectName("labelMuted")
        device_row.addWidget(device_label)

        self._radio_mic = QRadioButton("Microfone")
        self._radio_mic.setChecked(True)
        self._radio_mic.toggled.connect(self._on_source_changed)
        device_row.addWidget(self._radio_mic)

        self._radio_loopback = QRadioButton("Áudio do Sistema")
        device_row.addWidget(self._radio_loopback)

        device_row.addSpacing(12)

        self._device_selector = DeviceSelector(mode="mic")
        device_row.addWidget(self._device_selector, 1)

        layout.addLayout(device_row)

        # ── status indicator ─────────────────────────────────────────────
        self._status_label = QLabel("○  Botão flutuante inativo")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._status_label.setObjectName("labelMuted")
        layout.addWidget(self._status_label)

        # ── history list ─────────────────────────────────────────────────
        history_label = QLabel("Histórico de injeções:")
        history_label.setObjectName("labelMuted")
        layout.addWidget(history_label)

        self._history_list = QListWidget()
        self._history_list.setObjectName("historyList")
        self._history_list.setStyleSheet(
            "QListWidget { background-color: #1d2021; border: 1px solid #3c3836;"
            " border-radius: 4px; color: #ebdbb2; font-size: 12px; }"
            " QListWidget::item { padding: 4px 8px; border-bottom: 1px solid #3c3836; }"
            " QListWidget::item:selected { background-color: #3c3836; }"
        )
        layout.addWidget(self._history_list, 1)

        # ── clear button ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_clear = QPushButton("×  Limpar histórico")
        self._btn_clear.setObjectName("btnClear")
        self._btn_clear.clicked.connect(self._history_list.clear)
        btn_row.addWidget(self._btn_clear)
        layout.addLayout(btn_row)

    # ── signal wiring ─────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._text_injected_sig.connect(self._on_text_injected_ui)

    # ── floating window lifecycle ─────────────────────────────────────────

    def activate(self) -> None:
        """Called when the tab is selected — creates and shows the floating window."""
        if self._floating_win is None:
            self._floating_win = FloatingWindow()
            self._floating_win.recording_toggled.connect(self._on_recording_toggled)
            self._floating_win.closed.connect(self._on_floating_closed)

        self._floating_win.show()
        self._update_status(active=True)
        logger.info("FloatingButtonPanel: floating window shown")

    def deactivate(self) -> None:
        """Called when the tab is deselected — hides the floating window and stops recording."""
        if self._is_recording:
            self._stop_recording()
        if self._floating_win:
            self._floating_win.hide()
        self._update_status(active=False)
        logger.info("FloatingButtonPanel: floating window hidden")

    def _on_floating_closed(self) -> None:
        """User closed the floating window — notify main window to restore itself."""
        if self._is_recording:
            self._stop_recording()
        self._floating_win = None
        self._update_status(active=False)

        # Ask the parent MainWindow to restore
        main_win = self.window()
        if main_win and hasattr(main_win, "on_floating_window_closed"):
            main_win.on_floating_window_closed()

    # ── recording ─────────────────────────────────────────────────────────

    def _on_recording_toggled(self, recording: bool) -> None:
        if recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        mode = "loopback" if self._radio_loopback.isChecked() else "mic"
        device_index = self._device_selector.selected_device_index()
        try:
            self._mode.start_mode_floating(
                device_index=device_index,
                mode=mode,
                on_segment=self._on_segment_ready,
            )
            # Reset injector first-space tracking for each new recording session
            self._reset_event.set()
            self._is_recording = True
            logger.info("Floating mode recording started (mode=%s, device=%s)", mode, device_index)
        except Exception:
            logger.exception("Failed to start floating mode")
            if self._floating_win:
                self._floating_win.set_recording(False)

    def _stop_recording(self) -> None:
        threading.Thread(
            target=self._stop_workers_thread,
            daemon=True,
            name="float-stop-thread",
        ).start()

    def _stop_workers_thread(self) -> None:
        self._mode.stop_mode_floating()
        QMetaObject.invokeMethod(self, "_finalize_stop", Qt.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def _finalize_stop(self) -> None:
        self._is_recording = False
        if self._floating_win:
            self._floating_win.set_recording(False)

    # ── segment callback (asyncio thread) ────────────────────────────────

    def _on_segment_ready(self, text: str) -> None:
        """Called from the transcription engine asyncio thread.
        Queues the text for injection — never blocks the event loop.
        """
        text = text.strip()
        if text:
            self._inject_queue.put(text)

    # ── injection worker (dedicated thread) ───────────────────────────────

    def _injection_worker(self) -> None:
        """Runs in its own thread. Consumes texts and injects them via pynput."""
        from app.modes.text_injector import TextInjector
        injector = TextInjector(append_newline=False)

        while True:
            # Check for a session reset (new recording started)
            if self._reset_event.is_set():
                self._reset_event.clear()
                injector = TextInjector(append_newline=False)  # fresh instance = fresh first-space

            try:
                text = self._inject_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            success = injector.inject(text)
            if success:
                self._text_injected_sig.emit(text)
            self._inject_queue.task_done()

    # ── Qt slot (main thread) ─────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_text_injected_ui(self, text: str) -> None:
        self._injected_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{timestamp}] {text}")
        self._history_list.addItem(item)
        self._history_list.scrollToBottom()

        if self._floating_win:
            self._floating_win.flash_injected()

    # ── helpers ───────────────────────────────────────────────────────────

    def _update_status(self, active: bool) -> None:
        if active:
            self._status_label.setText("● Botão flutuante ativo — clique nele para gravar")
            self._status_label.setStyleSheet("color: #b8bb26; font-weight: bold;")
        else:
            self._status_label.setText("○  Botão flutuante inativo")
            self._status_label.setStyleSheet("color: #a89984;")

    def _on_source_changed(self, _checked: bool) -> None:
        if self._radio_mic.isChecked():
            self._device_selector._mode = "mic"
        else:
            self._device_selector._mode = "loopback"
        self._device_selector.refresh()
