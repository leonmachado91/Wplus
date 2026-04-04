"""Main application window — Gruvbox dark themed."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QSystemTrayIcon,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from app.core.settings_manager import SettingsManager
    from app.core.transcript_buffer import TranscriptBuffer
    from app.server.server_manager import ServerManager

logger = logging.getLogger(__name__)

THEME_DIR = Path(__file__).parent / "theme"


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings: SettingsManager,
        server_manager: ServerManager,
        buffer: TranscriptBuffer,
        mode_controller: Any,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._server_manager = server_manager
        self._buffer = buffer
        self._mode_controller = mode_controller

        self.setWindowTitle("Transcription App")
        self.setMinimumSize(850, 600)

        self._load_theme()
        self._build_ui()
        self._setup_tray()
        self._update_status_bar()

        # wire file_watcher into REST API app.state (after build so panel exists)
        self._wire_file_watcher_to_rest()

        # periodic status bar refresh
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start(3000)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # header
        header = QWidget()
        header.setObjectName("header")
        header.setStyleSheet("background-color: #1d2021; padding: 4px 12px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)

        title = QLabel("Transcription App")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #83a598;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet("color: #b8bb26; font-size: 16px;")
        header_layout.addWidget(self._status_dot)

        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("labelStatus")
        header_layout.addWidget(self._status_label)

        # Settings button
        from PyQt6.QtWidgets import QToolButton
        self._btn_settings = QToolButton()
        self._btn_settings.setText("⚙")
        self._btn_settings.setStyleSheet("QToolButton { background: transparent; color: #a89984; font-size: 16px; border: none; padding-left: 8px; } QToolButton:hover { color: #fabd2f; }")
        self._btn_settings.clicked.connect(self._open_settings)
        header_layout.addWidget(self._btn_settings)

        layout.addWidget(header)

        # tabs
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # Live Transcription tab
        from app.ui.live_panel import LivePanel
        self._live_panel = LivePanel(self._settings, self._buffer, self._mode_controller)
        self._tabs.addTab(self._live_panel, "Live Transcription")

        # File Watcher tab (Mode 1)
        from app.ui.file_watcher_panel import FileWatcherPanel
        self._file_watcher_panel = FileWatcherPanel(self._settings)
        self._tabs.addTab(self._file_watcher_panel, "File Watcher")

        # status bar
        self.statusBar().setStyleSheet(
            "background-color: #1d2021; color: #a89984; border-top: 1px solid #3c3836;"
        )

    def _load_theme(self) -> None:
        qss_path = THEME_DIR / "gruvbox_dark.qss"
        if qss_path.exists():
            self.setStyleSheet(qss_path.read_text(encoding="utf-8"))
            logger.info("Theme loaded: %s", qss_path.name)

    def _wire_file_watcher_to_rest(self) -> None:
        """Injects the FileWatcherMode instance into FastAPI app.state for REST access."""
        try:
            rest = self._server_manager.rest
            if rest and rest._server:
                fastapi_app = rest._server.config.app
                fastapi_app.state.file_watcher = self._file_watcher_panel.watcher
                logger.info("File watcher wired to REST API app.state")
        except Exception:
            logger.debug("Could not wire file_watcher to REST state (server may still be starting)")

    def _update_status_bar(self) -> None:
        srv = self._settings.settings.server
        parts: list[str] = []
        if srv.websocket_enabled:
            ws_clients = self._server_manager.ws.client_count if self._server_manager.ws else 0
            parts.append(f"WS: ws://{srv.websocket_host}:{srv.websocket_port} ●  Clients: {ws_clients}")
        if srv.rest_api_enabled:
            parts.append(f"REST: http://127.0.0.1:{srv.rest_api_port} ●")
        self.statusBar().showMessage("   ".join(parts))

    # ── system tray ──────────────────────────────────────────────────────

    def _setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("System tray not available")
            return

        self._tray = QSystemTrayIcon(self)
        self._tray.setToolTip("Transcription App")

        app_icon = QApplication.instance().windowIcon()
        if not app_icon.isNull():
            self._tray.setIcon(app_icon)

        menu = QMenu()
        show_action = QAction("Show / Hide", self)
        show_action.triggered.connect(self._toggle_visibility)
        menu.addAction(show_action)

        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self._open_settings)
        menu.addAction(settings_action)

        menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.instance().quit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.activateWindow()

    def _open_settings(self) -> None:
        from app.ui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self._settings, self)
        dialog.exec()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._toggle_visibility()

    # ── overrides ────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._settings.settings.ui.minimize_to_tray and hasattr(self, "_tray"):
            event.ignore()
            self.hide()
        else:
            event.accept()
