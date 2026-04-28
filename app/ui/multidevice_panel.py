"""Multi-device session panel — PyQt6 tab for multi-participant transcription."""
from __future__ import annotations

import logging
import socket
import threading
import time
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.ui.participant_list_widget import ParticipantListWidget
from app.ui.qr_widget import QRWidget
from app.ui.transcript_widget import TranscriptWidget

if TYPE_CHECKING:
    from app.core.settings_manager import SettingsManager
    from app.core.transcript_buffer import TranscriptBuffer
    from app.server.server_manager import ServerManager

logger = logging.getLogger(__name__)


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class MultidevicePanel(QWidget):
    """Tab: 'Multi-Dispositivo' — start a session, show QR, list participants."""

    _sig_participant_update = pyqtSignal()    # cross-thread → refresh table
    _sig_status_update = pyqtSignal(str)      # cross-thread → update status label
    _sig_segment_received = pyqtSignal(dict)  # cross-thread → add segment to timeline
    _sig_segment_updated = pyqtSignal(str, dict)  # cross-thread → update segment

    def __init__(
        self,
        settings: "SettingsManager",
        server_manager: "ServerManager",
        buffer: "TranscriptBuffer",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._server_manager = server_manager
        self._buffer = buffer
        self._engine = None   # TranscriptionEngine created on session start
        self._manager: Optional[object] = None  # ParticipantManager, lazy import
        self._session_start_time: Optional[float] = None

        self._build_ui()

        # Cross-thread signal wiring
        self._sig_participant_update.connect(self._refresh_participants)
        self._sig_status_update.connect(self._lbl_status.setText)
        self._sig_segment_received.connect(self._transcript.add_segment)
        self._sig_segment_updated.connect(self._transcript.update_segment)

        # Periodic refresh of participant list and elapsed time
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._tick)
        self._refresh_timer.start(1500)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Top: session controls ────────────────────────────────────────
        ctrl_group = QGroupBox("Sessão Multi-Dispositivo")
        ctrl_group.setStyleSheet("QGroupBox { color: #83a598; font-weight: bold; }")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(8)

        # Session name row
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Nome da sessão:"))
        self._inp_name = QLineEdit()
        self._inp_name.setPlaceholderText("ex: reunião-planejamento")
        self._inp_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        name_row.addWidget(self._inp_name)
        ctrl_layout.addLayout(name_row)

        # Mode radio buttons
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Modo:"))
        self._mode_group = QButtonGroup(self)
        for label, value in [("Presencial", "presencial"), ("Auto", "auto"), ("Remoto", "remoto")]:
            rb = QRadioButton(label)
            rb.setProperty("mode_value", value)
            if value == "auto":
                rb.setChecked(True)
            self._mode_group.addButton(rb)
            mode_row.addWidget(rb)
        mode_row.addStretch()
        ctrl_layout.addLayout(mode_row)

        # Start / stop button + status
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Iniciar Sessão")
        self._btn_start.setFixedHeight(36)
        self._btn_start.setStyleSheet(
            "QPushButton { background: #83a598; color: #1d2021; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background: #a1c4b8; }"
            "QPushButton:disabled { background: #504945; color: #7c6f64; }"
        )
        self._btn_start.clicked.connect(self._toggle_session)
        btn_row.addWidget(self._btn_start)
        ctrl_layout.addLayout(btn_row)

        self._lbl_status = QLabel("Sessão inativa")
        self._lbl_status.setStyleSheet("color: #a89984; font-size: 12px;")
        ctrl_layout.addWidget(self._lbl_status)

        root.addWidget(ctrl_group)

        # ── Middle: QR + participants + BleedGate controls ───────────────
        mid_row = QHBoxLayout()
        mid_row.setSpacing(12)

        # QR code widget
        qr_group = QGroupBox("QR de Entrada")
        qr_group.setStyleSheet("QGroupBox { color: #83a598; font-weight: bold; }")
        qr_layout = QVBoxLayout(qr_group)
        self._qr_widget = QRWidget(size=160)
        qr_layout.addWidget(self._qr_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        self._lbl_url = QLabel()
        self._lbl_url.setWordWrap(True)
        self._lbl_url.setStyleSheet("color: #83a598; font-size: 10px; font-family: monospace;")
        self._lbl_url.setAlignment(Qt.AlignmentFlag.AlignCenter)
        qr_layout.addWidget(self._lbl_url)
        qr_group.setFixedWidth(200)
        mid_row.addWidget(qr_group)

        # Participant list
        part_group = QGroupBox("Participantes (0)")
        part_group.setStyleSheet("QGroupBox { color: #83a598; font-weight: bold; }")
        self._part_group = part_group
        part_layout = QVBoxLayout(part_group)
        self._part_list = ParticipantListWidget(
            on_mute=self._on_mute,
            on_rename=self._on_rename,
            on_kick=self._on_kick,
        )
        part_layout.addWidget(self._part_list)
        mid_row.addWidget(part_group, stretch=1)

        # BleedGate controls panel
        gate_group = QGroupBox("BleedGate — Controles")
        gate_group.setStyleSheet("QGroupBox { color: #d3869b; font-weight: bold; }")
        gate_layout = QVBoxLayout(gate_group)
        gate_layout.setSpacing(6)

        # Gate enabled toggle
        self._chk_gate = QCheckBox("BleedGate ativo")
        self._chk_gate.setStyleSheet("color: #d3869b;")
        self._chk_gate.toggled.connect(self._on_gate_toggle)
        gate_layout.addWidget(self._chk_gate)

        # TDOA enabled
        self._chk_tdoa = QCheckBox("Usar TDOA (timestamp físico)")
        self._chk_tdoa.setChecked(True)
        self._chk_tdoa.setToolTip(
            "TDOA: compara o momento de captura de cada dispositivo.\n"
            "O mic que captou o som primeiro é o mais próximo do falante."
        )
        self._chk_tdoa.toggled.connect(self._on_gate_param_changed)
        gate_layout.addWidget(self._chk_tdoa)

        # Separator
        sep = QLabel("─" * 30)
        sep.setStyleSheet("color: #504945; font-size: 10px;")
        gate_layout.addWidget(sep)

        # window_ms
        gate_layout.addWidget(self._make_label("Janela de agrupamento (ms):"))
        self._spin_window = QSpinBox()
        self._spin_window.setRange(50, 2000)
        self._spin_window.setValue(250)
        self._spin_window.setSuffix(" ms")
        self._spin_window.setToolTip(
            "Chunks com sobreposição dentro desse intervalo\n"
            "são avaliados juntos como um grupo."
        )
        self._spin_window.valueChanged.connect(self._on_gate_param_changed)
        gate_layout.addWidget(self._spin_window)

        # tdoa_min_ms
        gate_layout.addWidget(self._make_label("TDOA mínimo (ms):"))
        self._spin_tdoa_min = QSpinBox()
        self._spin_tdoa_min.setRange(5, 500)
        self._spin_tdoa_min.setValue(20)
        self._spin_tdoa_min.setSuffix(" ms")
        self._spin_tdoa_min.setToolTip(
            "Diferença mínima de timestamp para o TDOA\n"
            "ser considerado válido. Abaixo disso, usa RMS."
        )
        self._spin_tdoa_min.valueChanged.connect(self._on_gate_param_changed)
        gate_layout.addWidget(self._spin_tdoa_min)

        # margin_db
        gate_layout.addWidget(self._make_label("Margem RMS (dB, fallback):"))
        self._spin_margin = QDoubleSpinBox()
        self._spin_margin.setRange(0.0, 20.0)
        self._spin_margin.setValue(6.0)
        self._spin_margin.setSuffix(" dB")
        self._spin_margin.setSingleStep(0.5)
        self._spin_margin.setToolTip(
            "Quando TDOA não discrimina, o falante dominante\n"
            "precisa ter pelo menos X dB a mais que os outros."
        )
        self._spin_margin.valueChanged.connect(self._on_gate_param_changed)
        gate_layout.addWidget(self._spin_margin)

        gate_layout.addStretch()
        gate_group.setFixedWidth(210)
        mid_row.addWidget(gate_group)

        root.addLayout(mid_row)

        # ── Bottom: transcript timeline ──────────────────────────────────
        transcript_group = QGroupBox("Transcrição em Tempo Real")
        transcript_group.setStyleSheet("QGroupBox { color: #83a598; font-weight: bold; }")
        transcript_layout = QVBoxLayout(transcript_group)
        transcript_layout.setContentsMargins(4, 8, 4, 4)

        self._transcript = TranscriptWidget()
        transcript_layout.addWidget(self._transcript)

        root.addWidget(transcript_group, stretch=1)

    @staticmethod
    def _make_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #a89984; font-size: 11px;")
        return lbl

    # ── session start / stop ─────────────────────────────────────────────

    def _toggle_session(self) -> None:
        if self._manager and getattr(self._manager, "is_active", False):
            self._stop_session()
        else:
            self._start_session()

    def _start_session(self) -> None:
        from app.multidevice.participant_manager import ParticipantManager
        from app.transcription.groq_engine import TranscriptionEngine

        if not self._settings.get("api", "groq_api_key"):
            self._lbl_status.setText("⚠ Groq API key não configurada — vá em Configurações.")
            return

        # Dedicated engine for multi-device mode (independent of live mode)
        if self._engine is None:
            self._engine = TranscriptionEngine(self._settings, self._buffer, sample_rate=16000)
            self._engine.start()

        manager = ParticipantManager(
            settings=self._settings,
            engine=self._engine,
            buffer=self._buffer,
        )
        manager.add_listener(self._on_manager_event)
        self._manager = manager

        # Wire manager to FastAPI app state so WebSocket router can find it
        self._wire_manager(manager)

        # Connect transcript buffer to timeline
        self._buffer.add_listener(self._on_buffer_event)
        self._transcript.clear_transcript()

        session_name = self._inp_name.text().strip() or "sessao"
        code = manager.create_session(session_name)

        # Activate bleed gate based on selected mode and sync UI checkbox
        selected_button = self._mode_group.checkedButton()
        if selected_button:
            selected_mode = selected_button.property("mode_value")
            gate_on = selected_mode == "presencial"
            manager.set_gate_enabled(gate_on)
            self._chk_gate.blockSignals(True)
            self._chk_gate.setChecked(gate_on)
            self._chk_gate.blockSignals(False)
            if gate_on:
                logger.info("Modo Presencial detectado: BleedGate ativado automaticamente.")
                # Apply current UI params to the newly created gate
                self._apply_gate_params()

        self._session_start_time = time.monotonic()
        self._btn_start.setText("■  Encerrar Sessão")

        port = self._settings.get("server", "rest_api_port")
        ip = _local_ip()
        url = f"http://{ip}:{port}/join/{code}"
        self._lbl_url.setText(url)

        # Fetch QR from our own endpoint in a background thread
        threading.Thread(target=self._fetch_qr, args=(code, url, port), daemon=True).start()

        logger.info("Multi-device session started: %s → %s", code, url)

    def _stop_session(self) -> None:
        self._buffer.remove_listener(self._on_buffer_event)

        if self._manager:
            self._manager.stop_session()
        self._manager = None
        if self._engine:
            self._engine.stop()
            self._engine = None
        self._session_start_time = None
        self._wire_manager(None)

        self._btn_start.setText("▶  Iniciar Sessão")
        self._lbl_status.setText("Sessão encerrada")
        self._qr_widget.clear()
        self._lbl_url.setText("")
        self._part_list.clear()
        self._part_group.setTitle("Participantes (0)")
        self._chk_gate.blockSignals(True)
        self._chk_gate.setChecked(False)
        self._chk_gate.blockSignals(False)

    def _wire_manager(self, manager) -> None:
        """Store (or clear) the participant manager on the FastAPI app state."""
        try:
            rest = self._server_manager.rest
            if rest:
                fastapi_app = rest.fastapi_app
                if fastapi_app is not None:
                    fastapi_app.state.participant_manager = manager
                else:
                    logger.warning("_wire_manager: REST API not started yet")
        except Exception as exc:
            logger.warning("Could not wire participant manager to REST: %s", exc)

    def _fetch_qr(self, code: str, url: str, port: int) -> None:
        """Download QR PNG from our own REST API and display it."""
        try:
            import urllib.request
            qr_url = f"http://127.0.0.1:{port}/join/{code}/qr"
            with urllib.request.urlopen(qr_url, timeout=5) as resp:
                png_bytes = resp.read()
            self._pending_qr = (png_bytes, url)
            QTimer.singleShot(0, self._apply_pending_qr)
        except Exception as exc:
            logger.warning("Could not fetch QR: %s", exc)

    def _apply_pending_qr(self) -> None:
        if hasattr(self, "_pending_qr"):
            png_bytes, url = self._pending_qr
            del self._pending_qr
            self._qr_widget.set_qr(png_bytes, url)

    # ── BleedGate control callbacks ──────────────────────────────────────

    def _on_gate_toggle(self, checked: bool) -> None:
        if self._manager:
            self._manager.set_gate_enabled(checked)
            if checked:
                self._apply_gate_params()

    def _on_gate_param_changed(self) -> None:
        """Called whenever any gate spinbox changes — apply immediately if gate exists."""
        if self._manager and getattr(self._manager, "gate_enabled", False):
            self._apply_gate_params()

    def _apply_gate_params(self) -> None:
        """Push current UI control values to the live BleedGateCoordinator."""
        if self._manager is None:
            return
        gate = getattr(self._manager, "_gate", None)
        if gate is None:
            return
        gate.update_params(
            window_ms=self._spin_window.value(),
            margin_db=self._spin_margin.value(),
            tdoa_min_ms=self._spin_tdoa_min.value(),
            tdoa_enabled=self._chk_tdoa.isChecked(),
        )

    # ── participant event callbacks ───────────────────────────────────────

    def _on_manager_event(self, event: str, data: dict) -> None:
        """Called from manager background threads — schedule UI update on main thread."""
        if event in ("participant_joined", "participant_left", "participant_muted", "participant_renamed"):
            self._sig_participant_update.emit()
        elif event == "session_created":
            self._sig_status_update.emit(f"Sessão ativa: {data.get('code')} — aguardando participantes")
        elif event == "session_stopped":
            self._sig_status_update.emit("Sessão encerrada")

    def _refresh_participants(self) -> None:
        if self._manager is None:
            return
        participants = self._manager.get_participants()
        self._part_list.update_participants(participants)
        self._part_group.setTitle(f"Participantes ({len(participants)})")

    def _on_mute(self, token: str, muted: bool) -> None:
        if self._manager:
            self._manager.mute_participant(token, muted)

    def _on_rename(self, token: str, new_name: str) -> None:
        if self._manager:
            self._manager.rename_participant(token, new_name)

    def _on_kick(self, token: str) -> None:
        if self._manager:
            self._manager.remove_participant(token)

    # ── TranscriptBuffer listener (any thread) ────────────────────────────

    def _on_buffer_event(self, event: str, data: dict) -> None:
        """Receives transcript events from the shared buffer — bridges to UI thread."""
        if event in ("segment_final", "segment_partial"):
            seg = data.get("segment", {})
            self._sig_segment_received.emit(seg)
        elif event == "segment_updated":
            seg = data.get("segment", {})
            seg_id = seg.get("id")
            if seg_id:
                self._sig_segment_updated.emit(seg_id, seg)

    # ── periodic tick ────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._manager and getattr(self._manager, "is_active", False) and self._session_start_time:
            elapsed = int(time.monotonic() - self._session_start_time)
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            code = self._manager.session_code or ""
            n = self._manager.participant_count
            self._lbl_status.setText(
                f"Sessão ativa: {code} — {n} participante(s) — {h:02d}:{m:02d}:{s:02d}"
            )
            # Also refresh participant chunk counts
            self._refresh_participants()
