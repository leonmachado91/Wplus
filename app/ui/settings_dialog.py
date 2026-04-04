"""Settings dialog for the application."""

from __future__ import annotations

from typing import cast, Any
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QStackedWidget,
    QDialogButtonBox, QWidget, QLabel, QLineEdit, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QGroupBox, QFormLayout, QToolButton, QPlainTextEdit
)

from app.core.settings_manager import SettingsManager

class SettingsDialog(QDialog):
    """A dialog window for managing application settings via a lateral menu."""
    
    settings_saved = pyqtSignal()

    def __init__(self, settings_manager: SettingsManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings_manager
        
        self.setWindowTitle("Settings")
        self.setMinimumSize(800, 500)
        
        self._ui_elements: dict[str, dict[str, Any]] = {
            "api": {},
            "audio": {},
            "vad": {},
            "diarization": {},
            "server": {},
            "ui": {}
        }
        
        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Lateral Menu
        self._list_widget = QListWidget()
        self._list_widget.setFixedWidth(180)
        self._list_widget.currentRowChanged.connect(self._change_page)
        main_layout.addWidget(self._list_widget)

        # Main Content Area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self._stacked_widget = QStackedWidget()
        right_layout.addWidget(self._stacked_widget)
        
        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._save_and_accept)
        btn_box.rejected.connect(self.reject)
        right_layout.addWidget(btn_box)
        
        main_layout.addWidget(right_panel)

        # Build Pages
        self._add_page("API Keys", self._build_api_page())
        self._add_page("Audio", self._build_audio_page())
        self._add_page("VAD (Voice Activity)", self._build_vad_page())
        self._add_page("Diarization", self._build_diarization_page())
        self._add_page("Server & Network", self._build_server_page())
        self._add_page("Interface", self._build_interface_page())
        
        self._list_widget.setCurrentRow(0)

    def _add_page(self, title: str, widget: QWidget) -> None:
        self._list_widget.addItem(title)
        self._stacked_widget.addWidget(widget)

    def _change_page(self, index: int) -> None:
        self._stacked_widget.setCurrentIndex(index)

    # ── Input Helpers ────────────────────────────────────────────────────────
    
    def _create_password_input(self, group: str, key: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        line_edit = QLineEdit()
        line_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._ui_elements[group][key] = line_edit
        
        toggle_btn = QToolButton()
        toggle_btn.setText("👁")
        toggle_btn.setCheckable(True)
        toggle_btn.toggled.connect(
            lambda checked, le=line_edit: le.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        
        layout.addWidget(line_edit)
        layout.addWidget(toggle_btn)
        return container

    def _create_slider_input(self, group: str, key: str, min_val: float, max_val: float, val_multiplier: float = 100.0) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_val * val_multiplier))
        slider.setMaximum(int(max_val * val_multiplier))
        
        val_label = QLabel()
        val_label.setFixedWidth(40)
        
        def update_label(v: int) -> None:
            val_label.setText(f"{v / val_multiplier:.2f}")
            
        slider.valueChanged.connect(update_label)
        self._ui_elements[group][key] = (slider, val_multiplier)
        
        layout.addWidget(slider)
        layout.addWidget(val_label)
        return container

    # ── Build Pages ─────────────────────────────────────────────────────────

    def _build_api_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        layout.addRow(QLabel("<h2>Groq Configuration</h2>"))
        layout.addRow("API Key:", self._create_password_input("api", "groq_api_key"))
        
        model_cb = QComboBox()
        model_cb.addItems(["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"])
        self._ui_elements["api"]["groq_model"] = model_cb
        layout.addRow("Model:", model_cb)

        lang_input = QLineEdit()
        lang_input.setPlaceholderText("ex: pt, en, es (opcional)")
        self._ui_elements["api"]["groq_language"] = lang_input
        layout.addRow("Language:", lang_input)

        prompt_input = QLineEdit()
        prompt_input.setPlaceholderText("Contexto ou palavras específicas...")
        self._ui_elements["api"]["groq_prompt"] = prompt_input
        layout.addRow("Prompt:", prompt_input)
        
        temp_spin = QDoubleSpinBox()
        temp_spin.setRange(0.0, 2.0)
        temp_spin.setSingleStep(0.1)
        self._ui_elements["api"]["groq_temperature"] = temp_spin
        layout.addRow("Temperature:", temp_spin)
        
        layout.addRow(QLabel("<hr><h2>Hugging Face (For Diarization)</h2>"))
        layout.addRow("HF Token:", self._create_password_input("api", "huggingface_token"))
        
        return page

    def _build_audio_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        
        sr_cb = QComboBox()
        sr_cb.addItems(["16000", "24000", "44100", "48000"])
        self._ui_elements["audio"]["sample_rate"] = sr_cb
        layout.addRow("Target Sample Rate:", sr_cb)
        
        channels_cb = QComboBox()
        channels_cb.addItems(["1", "2"])
        self._ui_elements["audio"]["channels"] = channels_cb
        layout.addRow("Channels:", channels_cb)
        
        return page

    def _build_vad_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        
        enabled_cb = QCheckBox("Enable Voice Activity Detection")
        self._ui_elements["vad"]["enabled"] = enabled_cb
        layout.addRow("", enabled_cb)
        
        layout.addRow("Onset Threshold:", self._create_slider_input("vad", "onset_threshold", 0.1, 0.9))
        layout.addRow("Offset Threshold:", self._create_slider_input("vad", "offset_threshold", 0.1, 0.9))
        
        min_speech = QSpinBox()
        min_speech.setRange(50, 1000)
        min_speech.setSuffix(" ms")
        self._ui_elements["vad"]["min_speech_duration_ms"] = min_speech
        layout.addRow("Min Speech Duration:", min_speech)
        
        speech_pad = QSpinBox()
        speech_pad.setRange(50, 1000)
        speech_pad.setSuffix(" ms")
        self._ui_elements["vad"]["speech_pad_ms"] = speech_pad
        layout.addRow("Speech Pre-roll/Pad:", speech_pad)
        
        max_chunk = QSpinBox()
        max_chunk.setRange(5, 60)
        max_chunk.setSuffix(" s")
        self._ui_elements["vad"]["max_chunk_duration_s"] = max_chunk
        layout.addRow("Fallback Chunk Cutter:", max_chunk)
        
        return page

    def _build_diarization_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        
        enabled_cb = QCheckBox("Enable Speaker Diarization")
        self._ui_elements["diarization"]["enabled"] = enabled_cb
        layout.addRow("", enabled_cb)
        
        info_label = QLabel(
            "<b>Warning:</b> First run will download ~2GB of models.<br>"
            "Ensure you provided your HuggingFace Token in the API tab<br>"
            "and accepted the terms for pyannote/speaker-diarization-3.1."
        )
        info_label.setStyleSheet("color: #fabd2f;")
        layout.addRow("", info_label)

        layout.addRow("Voice Similarity Threshold:", self._create_slider_input("diarization", "similarity_threshold", 0.1, 0.95))
        
        # Custom Labels Editor
        self._speaker_text_edit = QPlainTextEdit()
        self._speaker_text_edit.setPlaceholderText("SPEAKER_00=Leandro\nSPEAKER_01=Convidadão")
        self._speaker_text_edit.setMinimumHeight(100)
        layout.addRow(QLabel("Custom Speaker Names:<br><small>(One per line, format: ID=Name)</small>"), self._speaker_text_edit)
        
        return page

    def _build_server_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        
        ws_cb = QCheckBox("Enable WebSocket Server")
        self._ui_elements["server"]["websocket_enabled"] = ws_cb
        layout.addRow("", ws_cb)
        
        ws_port = QSpinBox()
        ws_port.setRange(1024, 65535)
        self._ui_elements["server"]["websocket_port"] = ws_port
        layout.addRow("WS Port:", ws_port)
        
        rest_cb = QCheckBox("Enable REST API Server")
        self._ui_elements["server"]["rest_api_enabled"] = rest_cb
        layout.addRow("", rest_cb)
        
        rest_port = QSpinBox()
        rest_port.setRange(1024, 65535)
        self._ui_elements["server"]["rest_api_port"] = rest_port
        layout.addRow("REST Port:", rest_port)
        
        return page

    def _build_interface_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        
        theme_cb = QComboBox()
        theme_cb.addItems(["gruvbox_dark", "light", "dark"])
        self._ui_elements["ui"]["theme"] = theme_cb
        layout.addRow("Theme:", theme_cb)
        
        tray_cb = QCheckBox("Minimize to system tray instead of closing")
        self._ui_elements["ui"]["minimize_to_tray"] = tray_cb
        layout.addRow("", tray_cb)
        
        return page

    # ── Load and Save ───────────────────────────────────────────────────────

    def _load_values(self) -> None:
        """Populate the UI forms with current settings."""
        settings_dict = self._settings.to_dict()
        
        # Load Special Elements
        diar = settings_dict.get("diarization", {})
        labels = diar.get("speaker_labels", {})
        lines = [f"{k}={v}" for k, v in labels.items()]
        self._speaker_text_edit.setPlainText("\n".join(lines))
        
        for group, items in self._ui_elements.items():
            if group not in settings_dict:
                continue
            
            group_data = settings_dict[group]
            for key, widget in items.items():
                if key not in group_data:
                    continue
                
                val = group_data[key]
                if isinstance(widget, QLineEdit):
                    widget.setText(str(val or ""))
                elif isinstance(widget, QComboBox):
                    idx = widget.findText(str(val))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                    else:
                        widget.setCurrentText(str(val))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(val))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(val))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(val))
                elif isinstance(widget, tuple) and isinstance(widget[0], QSlider):
                    slider, multiplier = widget
                    slider.setValue(int(float(val) * multiplier))

    def _save_and_accept(self) -> None:
        """Gather values from widgets and save them into the manager."""
        for group, items in self._ui_elements.items():
            updates = {}
            for key, widget in items.items():
                if isinstance(widget, QLineEdit):
                    updates[key] = widget.text()
                elif isinstance(widget, QComboBox):
                    updates[key] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    updates[key] = widget.isChecked()
                elif isinstance(widget, QSpinBox):
                    updates[key] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    updates[key] = widget.value()
                elif isinstance(widget, tuple) and isinstance(widget[0], QSlider):
                    slider, multiplier = widget
                    updates[key] = slider.value() / multiplier

            self._settings.update_section(group, updates)
            
        # Parse Special Elements
        lines = self._speaker_text_edit.toPlainText().strip().split("\n")
        speaker_map = {}
        for line in lines:
            if "=" in line:
                k, v = line.split("=", 1)
                speaker_map[k.strip()] = v.strip()
        self._settings.update_section("diarization", {"speaker_labels": speaker_map})
            
        self._settings.save_now()
        self.settings_saved.emit()
        self.accept()
