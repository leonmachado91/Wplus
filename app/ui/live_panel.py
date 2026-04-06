"""Live transcription panel — main UI for Mode 2 (mic/loopback recording)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.device_selector import DeviceSelector
from app.ui.transcript_widget import TranscriptWidget

if TYPE_CHECKING:
    from app.core.settings_manager import SettingsManager
    from app.core.transcript_buffer import TranscriptBuffer

logger = logging.getLogger(__name__)


class LivePanel(QWidget):
    """Full live transcription panel with controls, transcript display, and status."""

    # signals for thread-safe UI updates
    segment_received = pyqtSignal(dict)
    segment_updated = pyqtSignal(str, dict)
    status_changed = pyqtSignal(str)
    rms_updated = pyqtSignal(float)
    vad_state_changed = pyqtSignal(bool)
    diarization_status = pyqtSignal(str)  # thread-safe bridge for diarization messages

    def __init__(
        self,
        settings: SettingsManager,
        buffer: TranscriptBuffer,
        mode_controller: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._buffer = buffer
        self._mode = mode_controller

        self._bridge_thread: Optional[threading.Thread] = None
        self._is_recording = False

        self._build_ui()
        self._connect_signals()

        # RMS polling timer
        self._rms_timer = QTimer(self)
        self._rms_timer.timeout.connect(self._poll_rms)

        # Register as diarization status listener so messages appear in the status bar
        self._mode.on_diarization_status = self._on_diarization_status_msg

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # ── source + device row ──────────────────────────────────────────
        source_row = QHBoxLayout()
        source_row.setSpacing(12)

        source_label = QLabel("Fonte:")
        source_label.setObjectName("labelMuted")
        source_row.addWidget(source_label)

        self._chk_mic = QCheckBox("Microfone")
        self._chk_mic.setChecked(True)
        self._chk_mic.toggled.connect(self._on_source_changed)
        source_row.addWidget(self._chk_mic)

        self._chk_loopback = QCheckBox("Áudio do Sistema")
        self._chk_loopback.setChecked(False)
        self._chk_loopback.toggled.connect(self._on_source_changed)
        source_row.addWidget(self._chk_loopback)

        source_row.addSpacing(12)

        device_label = QLabel("Dispositivo do mic:")
        device_label.setObjectName("labelMuted")
        self._lbl_device = device_label
        source_row.addWidget(device_label)

        self._device_selector = DeviceSelector(mode="mic")
        source_row.addWidget(self._device_selector, 1)

        layout.addLayout(source_row)

        # ── echo warning (shown only when both sources are active) ─────────────
        self._lbl_echo_warn = QLabel(
            "⚠ Modo Ambos ativo: use fones de ouvido para evitar duplicação — "
            "o mic pode captar o áudio dos alto-falantes."
        )
        self._lbl_echo_warn.setObjectName("labelMuted")
        self._lbl_echo_warn.setStyleSheet("color: #fabd2f; font-size: 11px;")
        self._lbl_echo_warn.setVisible(False)
        layout.addWidget(self._lbl_echo_warn)

        # ── control buttons row ──────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_record = QPushButton("⏺  Gravar")
        self._btn_record.setObjectName("btnRecord")
        self._btn_record.setCheckable(True)
        self._btn_record.setMinimumWidth(90)
        self._btn_record.clicked.connect(self._on_record_clicked)
        btn_row.addWidget(self._btn_record)

        self._btn_pause = QPushButton("⏸  Pausar")
        self._btn_pause.setObjectName("btnPause")
        self._btn_pause.setCheckable(True)
        self._btn_pause.setEnabled(False)
        self._btn_pause.setMinimumWidth(90)
        self._btn_pause.clicked.connect(self._on_pause_clicked)
        btn_row.addWidget(self._btn_pause)

        self._btn_stop = QPushButton("⏹  Parar")
        self._btn_stop.setObjectName("btnStop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self._btn_stop)

        self._btn_export = QPushButton("↓  Exportar")
        self._btn_export.setObjectName("btnExport")
        self._btn_export.clicked.connect(self._on_export_clicked)
        btn_row.addWidget(self._btn_export)

        self._btn_clear = QPushButton("×  Limpar")
        self._btn_clear.setObjectName("btnClear")
        self._btn_clear.clicked.connect(self._on_clear_clicked)
        btn_row.addWidget(self._btn_clear)

        btn_row.addStretch()

        # salvamento automático
        self._chk_autosave = QCheckBox("Salvar automaticamente:")
        self._chk_autosave.toggled.connect(self._on_autosave_toggled)
        btn_row.addWidget(self._chk_autosave)

        self._txt_autosave_path = QLineEdit()
        self._txt_autosave_path.setPlaceholderText("pasta de saída...")
        self._txt_autosave_path.setMinimumWidth(160)
        btn_row.addWidget(self._txt_autosave_path)

        self._btn_browse = QPushButton("Procurar")
        self._btn_browse.clicked.connect(self._on_browse_clicked)
        btn_row.addWidget(self._btn_browse)

        layout.addLayout(btn_row)

        # ── transcript area ──────────────────────────────────────────────
        self._transcript = TranscriptWidget()
        # speaker_mapper is None until recording starts — injected in _start_recording()
        layout.addWidget(self._transcript, 1)

        # ── status bar ───────────────────────────────────────────────────
        status_row = QHBoxLayout()
        status_row.setSpacing(16)

        audio_label = QLabel("Áudio:")
        audio_label.setObjectName("labelMuted")
        status_row.addWidget(audio_label)

        self._rms_bar = QProgressBar()
        self._rms_bar.setRange(0, 100)
        self._rms_bar.setValue(0)
        self._rms_bar.setTextVisible(False)
        self._rms_bar.setFixedHeight(12)
        self._rms_bar.setMinimumWidth(120)
        status_row.addWidget(self._rms_bar)

        self._vad_label = QLabel("VAD: —")
        self._vad_label.setObjectName("labelMuted")
        self._vad_label.setMinimumWidth(100)
        status_row.addWidget(self._vad_label)

        self._status_label = QLabel("")
        self._status_label.setObjectName("labelMuted")
        status_row.addWidget(self._status_label, 1)

        layout.addLayout(status_row)

    # ── signal wiring ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.segment_received.connect(self._on_segment_received)
        self.segment_updated.connect(self._on_segment_updated)
        self.rms_updated.connect(self._on_rms_updated)
        self.vad_state_changed.connect(self._on_vad_state_changed)
        self.diarization_status.connect(self._on_diarization_status_ui)

    # ── recording lifecycle ──────────────────────────────────────────────

    def _on_record_clicked(self) -> None:
        if self._is_recording:
            return
        self._start_recording()

    def _on_stop_clicked(self) -> None:
        self._stop_recording()

    def _get_capture_mode(self) -> str:
        """Derive capture mode string from the current checkbox state."""
        mic = self._chk_mic.isChecked()
        loopback = self._chk_loopback.isChecked()
        if mic and loopback:
            return "both"
        if loopback:
            return "loopback"
        return "mic"  # default / only-mic

    def _start_recording(self) -> None:
        # Guard: at least one source must be selected
        if not self._chk_mic.isChecked() and not self._chk_loopback.isChecked():
            self._status_label.setText("⚠ Selecione ao menos uma fonte de áudio.")
            self._status_label.setStyleSheet("color: #fabd2f;")
            self._btn_record.setChecked(False)
            return

        mode = self._get_capture_mode()
        # Device index only matters when mic is part of the capture
        device_index = self._device_selector.selected_device_index() if self._chk_mic.isChecked() else None

        # start session
        self._buffer.start_session()

        # auto-save
        if self._chk_autosave.isChecked() and self._txt_autosave_path.text():
            folder = Path(self._txt_autosave_path.text())
            path = folder / f"transcription_{self._buffer.session_id[:8]}.md"
            self._buffer.set_auto_save(path)

        # register buffer listener for UI updates
        self._buffer.add_listener(self._on_buffer_event)

        # build and start pipeline via ModeController
        try:
            self._mode.start_mode_live(device_index, mode)

            # Inject the speaker mapper now that the pipeline has created it
            self._transcript.set_speaker_mapper(self._mode.speaker_mapper)

            self._is_recording = True
            self._btn_record.setChecked(True)
            self._btn_stop.setEnabled(True)
            self._btn_pause.setEnabled(True)
            self._btn_pause.setChecked(False)
            self._btn_pause.setText("⏸  Pausar")
            self._btn_record.setEnabled(False)
            self._chk_mic.setEnabled(False)
            self._chk_loopback.setEnabled(False)
            self._device_selector.setEnabled(False)
            self._rms_timer.start(50)  # 20fps RMS polling

            mode_label = {"mic": "Mic", "loopback": "Sistema", "both": "Mic + Sistema"}.get(mode, mode)
            self._status_label.setText(f"Gravando ({mode_label})...")
            self._status_label.setStyleSheet("color: #fb4934; font-weight: bold;")
            logger.info("Recording started (mode=%s, device=%s)", mode, device_index)

        except Exception as e:
            logger.exception("Failed to start recording")
            self._status_label.setText(f"Erro: {e}")
            self._status_label.setStyleSheet("color: #fb4934;")
            self._cleanup_pipeline()

    def _cleanup_pipeline(self) -> None:
        """Reset UI state after a failed or interrupted start."""
        self._rms_timer.stop()
        self._rms_bar.setValue(0)
        self._is_recording = False
        self._btn_record.setChecked(False)
        self._btn_record.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._chk_mic.setEnabled(True)
        self._chk_loopback.setEnabled(True)
        self._device_selector.setEnabled(self._chk_mic.isChecked())

    def _stop_recording(self) -> None:
        if not self._is_recording:
            return

        self._status_label.setText("Parando...")
        self._status_label.setStyleSheet("color: #fabd2f;")
        self._btn_stop.setEnabled(False)

        # Offload the blocking .stop_mode_live() calls to a background thread
        threading.Thread(target=self._stop_workers_thread, daemon=True, name="panel-stop-thread").start()

    def _stop_workers_thread(self) -> None:
        self._mode.stop_mode_live()

        # Safely return to Main UI Thread
        from PyQt6.QtCore import QMetaObject, Qt
        QMetaObject.invokeMethod(self, "_finalize_stop", Qt.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def _finalize_stop(self) -> None:
        self._buffer.remove_listener(self._on_buffer_event)
        info = self._buffer.stop_session()
        self._buffer.set_auto_save(None)

        self._rms_timer.stop()
        self._rms_bar.setValue(0)
        self._vad_label.setText("VAD: —")  # reset

        self._is_recording = False
        self._btn_record.setChecked(False)
        self._btn_record.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._chk_mic.setEnabled(True)
        self._chk_loopback.setEnabled(True)
        self._device_selector.setEnabled(self._chk_mic.isChecked())

        count = info.get("segment_count", 0)
        dur = info.get("duration_s", 0.0)
        self._status_label.setText(f"Concluído — {count} segmentos, {dur:.1f}s")
        self._status_label.setStyleSheet("color: #b8bb26;")
        logger.info("Recording stopped: %d segments, %.1fs", count, dur)



    # ── buffer listener (called from any thread) ─────────────────────────

    def _on_buffer_event(self, event: str, data: dict) -> None:
        if event in ("segment_final", "segment_partial"):
            seg = data.get("segment", {})
            self.segment_received.emit(seg)
        elif event == "segment_updated":
            seg = data.get("segment", {})
            seg_id = seg.get("id")
            if seg_id:
                self.segment_updated.emit(seg_id, seg)

    # ── Qt slots (main thread) ───────────────────────────────────────────

    @pyqtSlot(dict)
    def _on_segment_received(self, segment: dict) -> None:
        self._transcript.add_segment(segment)

    @pyqtSlot(str, dict)
    def _on_segment_updated(self, seg_id: str, updates: dict) -> None:
        self._transcript.update_segment(seg_id, updates)

    @pyqtSlot(float)
    def _on_rms_updated(self, rms: float) -> None:
        if self._btn_pause.isChecked():
            return
        level = min(int(rms * 500), 100)  # scale for visibility
        self._rms_bar.setValue(level)

    @pyqtSlot(bool)
    def _on_vad_state_changed(self, is_speech: bool) -> None:
        if is_speech:
            self._vad_label.setText("VAD: FALA")
            self._vad_label.setStyleSheet("color: #b8bb26; font-weight: bold;")
        else:
            self._vad_label.setText("VAD: silêncio")
            self._vad_label.setStyleSheet("color: #a89984;")

    # ── diarization status (cross-thread safe) ───────────────────────────

    def _on_diarization_status_msg(self, message: str) -> None:
        """Called from diarization worker thread — bridge via signal."""
        self.diarization_status.emit(message)

    @pyqtSlot(str)
    def _on_diarization_status_ui(self, message: str) -> None:
        """Show diarization status in the panel's status label (runs on UI thread)."""
        is_error = any(w in message.lower() for w in ("erro", "error", "403", "acesso negado"))
        color = "#fb4934" if is_error else "#fabd2f"
        prefix = "🔈 " if not is_error else "⚠ "
        self._status_label.setText(f"{prefix}{message[:120]}")
        self._status_label.setStyleSheet(f"color: {color};")

    # ── RMS polling ──────────────────────────────────────────────────────

    def _poll_rms(self) -> None:
        """Read the latest RMS value from the capture engine's dedicated display queue.

        AudioCaptureEngine publishes float RMS values to rms_queue without blocking
        the VAD pipeline. We take all pending values and use the last one for display,
        keeping the main audio queue completely untouched.
        """
        if not self._is_recording or not self._mode.capture_engine or not self._mode.capture_engine.is_running:
            return
        try:
            rms_q = self._mode.capture_engine.rms_queue
            latest_rms = None
            while True:
                try:
                    latest_rms = rms_q.get_nowait()
                except Exception:
                    break

            if latest_rms is not None:
                self.rms_updated.emit(latest_rms)

            if self._mode.vad_processor:
                self.vad_state_changed.emit(self._mode.vad_processor.is_speech)
        except Exception:
            pass

    # ── button handlers ──────────────────────────────────────────────────

    def _on_source_changed(self, _checked: bool) -> None:
        """Update device selector visibility/mode when checkboxes change."""
        mic_on = self._chk_mic.isChecked()
        loopback_on = self._chk_loopback.isChecked()

        # Device selector is only relevant when mic is part of the capture.
        self._device_selector.setEnabled(mic_on)
        self._lbl_device.setEnabled(mic_on)
        if mic_on:
            self._device_selector.set_mode("mic")

        # Show echo warning only when both sources are active simultaneously.
        self._lbl_echo_warn.setVisible(mic_on and loopback_on)

        # Disable Record button if nothing is selected
        nothing_selected = not mic_on and not loopback_on
        self._btn_record.setEnabled(not nothing_selected and not self._is_recording)

    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Transcrição",
            "",
            "Markdown (*.md);;Texto (*.txt);;Todos os arquivos (*)",
        )
        if not path:
            return
        if path.endswith(".txt"):
            self._buffer.export_text(path)
        else:
            self._buffer.export_markdown(path)
        self._status_label.setText(f"Exportado: {Path(path).name}")
        self._status_label.setStyleSheet("color: #b8bb26;")

    @pyqtSlot(bool)
    def _on_pause_clicked(self, checked: bool) -> None:
        if not self._is_recording:
            return
        
        self._mode.set_paused(checked)
        if checked:
            self._btn_pause.setText("▶  Retomar")
            self._status_label.setText("Pausado (áudio ignorado)")
            self._status_label.setStyleSheet("color: #fabd2f;")
            self._btn_record.setText("⏸  Pausado")
            self._rms_bar.setValue(0)
        else:
            self._btn_pause.setText("⏸  Pausar")
            self._status_label.setText("Gravando (escutando)...")
            self._status_label.setStyleSheet("color: #fb4934; font-weight: bold;")
            self._btn_record.setText("⏺  Gravando...")

    def _on_clear_clicked(self) -> None:
        self._buffer.clear()
        self._transcript.clear_transcript()
        self._status_label.setText("Limpo")
        self._status_label.setStyleSheet("color: #a89984;")

    def _on_autosave_toggled(self, checked: bool) -> None:
        self._txt_autosave_path.setEnabled(checked)
        self._btn_browse.setEnabled(checked)

    def _on_browse_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Salvamento")
        if folder:
            self._txt_autosave_path.setText(folder)
