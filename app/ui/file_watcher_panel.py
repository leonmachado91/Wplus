"""File Watcher panel — Mode 1 UI tab."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QMetaObject
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QAbstractItemView,
)

from app.core.settings_manager import SettingsManager
from app.modes.file_watcher_mode import FileWatcherMode, FileJob

logger = logging.getLogger(__name__)


class FileWatcherPanel(QWidget):
    """UI panel for controlling the file watcher mode."""

    job_updated = pyqtSignal(object)  # FileJob

    def __init__(self, settings: SettingsManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._watcher = FileWatcherMode(settings)
        self._watcher.add_listener(self._on_job_update)

        self._build_ui()
        self._load_settings()

        # signal for cross-thread UI update
        self.job_updated.connect(self._refresh_table)

    @property
    def watcher(self) -> FileWatcherMode:
        return self._watcher

    # ── build UI ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── folders group ────────────────────────────────────────────────
        folder_group = QGroupBox("Pastas")
        folder_layout = QFormLayout(folder_group)

        # pasta monitorada
        watch_row = QHBoxLayout()
        self._txt_watch = QLineEdit()
        self._txt_watch.setPlaceholderText("Selecione a pasta a monitorar por arquivos de áudio...")
        self._btn_browse_watch = QPushButton("Procurar")
        self._btn_browse_watch.setFixedWidth(80)
        self._btn_browse_watch.clicked.connect(lambda: self._browse("watch"))
        watch_row.addWidget(self._txt_watch)
        watch_row.addWidget(self._btn_browse_watch)
        watch_container = QWidget()
        watch_container.setLayout(watch_row)
        folder_layout.addRow("Pasta Monitorada:", watch_container)

        # pasta de saída
        out_row = QHBoxLayout()
        self._txt_output = QLineEdit()
        self._txt_output.setPlaceholderText("Selecione a pasta de saída das transcrições...")
        self._btn_browse_out = QPushButton("Procurar")
        self._btn_browse_out.setFixedWidth(80)
        self._btn_browse_out.clicked.connect(lambda: self._browse("output"))
        out_row.addWidget(self._txt_output)
        out_row.addWidget(self._btn_browse_out)
        out_container = QWidget()
        out_container.setLayout(out_row)
        folder_layout.addRow("Pasta de Saída:", out_container)

        root.addWidget(folder_group)

        # ── controls ─────────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()

        self._btn_start = QPushButton("▶  Iniciar Monitoramento")
        self._btn_start.setObjectName("btn_start_watcher")
        self._btn_start.clicked.connect(self._on_start)

        self._btn_stop = QPushButton("■  Parar")
        self._btn_stop.setObjectName("btn_stop_watcher")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)

        self._status_label = QLabel("Aguardando")
        self._status_label.setStyleSheet("color: #a89984;")

        ctrl_row.addWidget(self._btn_start)
        ctrl_row.addWidget(self._btn_stop)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self._status_label)

        root.addLayout(ctrl_row)

        # ── job table ────────────────────────────────────────────────────
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Arquivo", "Progresso", "Status"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().resizeSection(1, 180)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().resizeSection(2, 120)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)

        root.addWidget(self._table)

        # map file_path → row index for updating
        self._row_map: dict[str, int] = {}

    # ── settings persistence ─────────────────────────────────────────────

    def _load_settings(self) -> None:
        cfg = self._settings.settings.mode1
        if cfg.watch_folder:
            self._txt_watch.setText(cfg.watch_folder)
        if cfg.output_folder:
            self._txt_output.setText(cfg.output_folder)

    def _save_folders(self) -> None:
        self._settings.update_section("mode1", {
            "watch_folder": self._txt_watch.text(),
            "output_folder": self._txt_output.text(),
        })

    # ── handlers ─────────────────────────────────────────────────────────

    def _browse(self, which: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, f"Select {which} folder")
        if not folder:
            return
        if which == "watch":
            self._txt_watch.setText(folder)
        else:
            self._txt_output.setText(folder)

    def _on_start(self) -> None:
        watch = self._txt_watch.text().strip()
        output = self._txt_output.text().strip()

        if not watch:
            self._status_label.setText("⚠ Selecione uma pasta para monitorar")
            self._status_label.setStyleSheet("color: #fb4934;")
            return
        if not output:
            self._status_label.setText("⚠ Selecione uma pasta de saída")
            self._status_label.setStyleSheet("color: #fb4934;")
            return

        self._save_folders()
        self._watcher.start(watch_folder=watch, output_folder=output)

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._txt_watch.setEnabled(False)
        self._txt_output.setEnabled(False)
        self._btn_browse_watch.setEnabled(False)
        self._btn_browse_out.setEnabled(False)

        self._status_label.setText("👁  Monitorando...")
        self._status_label.setStyleSheet("color: #b8bb26;")

    def _on_stop(self) -> None:
        # watcher.stop() joins threads (up to 8s) — must not block the UI thread
        self._btn_stop.setEnabled(False)
        self._status_label.setText("Parando…")
        self._status_label.setStyleSheet("color: #fabd2f;")
        threading.Thread(target=self._stop_worker, daemon=True, name="fw-stop").start()

    def _stop_worker(self) -> None:
        self._watcher.stop()
        QMetaObject.invokeMethod(self, "_finalize_stop", Qt.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def _finalize_stop(self) -> None:
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._txt_watch.setEnabled(True)
        self._txt_output.setEnabled(True)
        self._btn_browse_watch.setEnabled(True)
        self._btn_browse_out.setEnabled(True)
        self._status_label.setText("Parado")
        self._status_label.setStyleSheet("color: #a89984;")

    # ── job listener (called from worker thread) ─────────────────────────

    def _on_job_update(self, job: FileJob) -> None:
        # emit signal so update happens on Qt main thread
        self.job_updated.emit(job)

    @pyqtSlot(object)
    def _refresh_table(self, job: FileJob) -> None:
        key = str(job.file_path)

        if key not in self._row_map:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._row_map[key] = row

            # file name
            self._table.setItem(row, 0, QTableWidgetItem(job.file_path.name))

            # progress bar
            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            self._table.setCellWidget(row, 1, pbar)

            # status
            self._table.setItem(row, 2, QTableWidgetItem("queued"))
        
        row = self._row_map[key]

        # update progress
        pbar = self._table.cellWidget(row, 1)
        if isinstance(pbar, QProgressBar):
            pbar.setValue(int(job.progress * 100))

        # update status
        status_item = self._table.item(row, 2)
        if status_item:
            label = job.status
            if job.status == "error":
                label = f"✗ {job.error_msg[:30]}"
            elif job.status == "done":
                label = "✓ Concluído"
            elif job.status == "processing":
                label = "⏳ Processando"
            status_item.setText(label)

        self._table.scrollToBottom()
