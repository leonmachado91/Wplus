"""Diálogo de configurações da aplicação."""

from __future__ import annotations

from typing import cast, Any
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QStackedWidget,
    QDialogButtonBox, QWidget, QLabel, QLineEdit, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QGroupBox, QFormLayout, QToolButton, QPlainTextEdit
)

from app.core.settings_manager import SettingsManager


def _tip(widget: QWidget, text: str) -> QWidget:
    """Aplica tooltip e retorna o próprio widget."""
    widget.setToolTip(text)
    return widget


class SettingsDialog(QDialog):
    """Janela de configurações com menu lateral."""

    settings_saved = pyqtSignal()

    def __init__(self, settings_manager: SettingsManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings_manager

        self.setWindowTitle("Configurações")
        self.setMinimumSize(800, 500)

        self._ui_elements: dict[str, dict[str, Any]] = {
            "api": {},
            "audio": {},
            "vad": {},
            "diarization": {},
            "server": {},
            "ui": {},
            "filters": {},
        }

        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Menu lateral
        self._list_widget = QListWidget()
        self._list_widget.setFixedWidth(180)
        self._list_widget.currentRowChanged.connect(self._change_page)
        main_layout.addWidget(self._list_widget)

        # Área de conteúdo
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._stacked_widget = QStackedWidget()
        right_layout.addWidget(self._stacked_widget)

        # Botões OK / Cancelar
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setText("Salvar")
        btn_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        btn_box.accepted.connect(self._save_and_accept)
        btn_box.rejected.connect(self.reject)
        right_layout.addWidget(btn_box)

        main_layout.addWidget(right_panel)

        # Páginas
        self._add_page("Chaves de API", self._build_api_page())
        self._add_page("Áudio", self._build_audio_page())
        self._add_page("VAD (Detectar Voz)", self._build_vad_page())
        self._add_page("Diarização", self._build_diarization_page())
        self._add_page("Servidor & Rede", self._build_server_page())
        self._add_page("Interface", self._build_interface_page())
        self._add_page("Filtros de Ruído", self._build_filters_page())

        self._list_widget.setCurrentRow(0)

    def _add_page(self, title: str, widget: QWidget) -> None:
        self._list_widget.addItem(title)
        self._stacked_widget.addWidget(widget)

    def _change_page(self, index: int) -> None:
        self._stacked_widget.setCurrentIndex(index)

    # ── Helpers de input ─────────────────────────────────────────────────

    def _create_password_input(self, group: str, key: str, tooltip: str = "") -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        line_edit = QLineEdit()
        line_edit.setEchoMode(QLineEdit.EchoMode.Password)
        if tooltip:
            line_edit.setToolTip(tooltip)
        self._ui_elements[group][key] = line_edit

        toggle_btn = QToolButton()
        toggle_btn.setText("👁")
        toggle_btn.setCheckable(True)
        toggle_btn.setToolTip("Mostrar / ocultar")
        toggle_btn.toggled.connect(
            lambda checked, le=line_edit: le.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )

        layout.addWidget(line_edit)
        layout.addWidget(toggle_btn)
        return container

    def _create_slider_input(
        self,
        group: str,
        key: str,
        min_val: float,
        max_val: float,
        val_multiplier: float = 100.0,
        tooltip: str = "",
    ) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_val * val_multiplier))
        slider.setMaximum(int(max_val * val_multiplier))
        if tooltip:
            slider.setToolTip(tooltip)
            container.setToolTip(tooltip)

        val_label = QLabel()
        val_label.setFixedWidth(40)

        def update_label(v: int) -> None:
            val_label.setText(f"{v / val_multiplier:.2f}")

        slider.valueChanged.connect(update_label)
        self._ui_elements[group][key] = (slider, val_multiplier)

        layout.addWidget(slider)
        layout.addWidget(val_label)
        return container

    # ── Páginas ──────────────────────────────────────────────────────────

    def _build_api_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addRow(QLabel("<h2>Configuração Groq</h2>"))

        layout.addRow(
            "Chave de API:",
            self._create_password_input(
                "api", "groq_api_key",
                tooltip="Chave secreta para autenticação na API do Groq. Obtenha em console.groq.com."
            )
        )

        model_cb = QComboBox()
        model_cb.addItems(["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"])
        model_cb.setToolTip(
            "Modelo Whisper usado para transcrição.\n"
            "• whisper-large-v3-turbo — melhor custo-benefício (recomendado)\n"
            "• whisper-large-v3 — máxima precisão, mais lento\n"
            "• distil-whisper-large-v3-en — apenas inglês, velocidade máxima"
        )
        self._ui_elements["api"]["groq_model"] = model_cb
        layout.addRow("Modelo:", model_cb)

        lang_input = QLineEdit()
        lang_input.setPlaceholderText("ex: pt, en, es (opcional)")
        lang_input.setToolTip(
            "Código ISO 639-1 do idioma falado no áudio.\n"
            "Deixe em branco para detecção automática.\n"
            "Exemplos: pt (português), en (inglês), es (espanhol)."
        )
        self._ui_elements["api"]["groq_language"] = lang_input
        layout.addRow("Idioma:", lang_input)

        prompt_input = QLineEdit()
        prompt_input.setPlaceholderText("Contexto ou palavras específicas...")
        prompt_input.setToolTip(
            "Texto de contexto enviado ao Whisper antes de cada chunk.\n"
            "Use para corrigir nomes próprios, termos técnicos ou gírias.\n"
            "Exemplo: 'Reunião sobre o projeto WPlus. Participantes: Leonardo, Ana.'"
        )
        self._ui_elements["api"]["groq_prompt"] = prompt_input
        layout.addRow("Prompt de Contexto:", prompt_input)

        rolling_cb = QCheckBox("Habilitar Rolling Context")
        rolling_cb.setToolTip(
            "Se ativado, o aplicativo enviará partes da transcrição recente \n"
            "junto com o prompt para ajudar o modelo a manter o contexto da conversa.\n"
            "Desative se quiser que cada trecho de áudio seja transcrito do zero absoluto."
        )
        self._ui_elements["api"]["use_rolling_context"] = rolling_cb
        layout.addRow("Contexto de Histórico:", rolling_cb)

        temp_spin = QDoubleSpinBox()
        temp_spin.setRange(0.0, 2.0)
        temp_spin.setSingleStep(0.1)
        temp_spin.setToolTip(
            "Controla a 'criatividade' do modelo na transcrição.\n"
            "• 0.0 — determinístico (recomendado para transcrição)\n"
            "• Valores altos aumentam variações mas reduzem consistência."
        )
        self._ui_elements["api"]["groq_temperature"] = temp_spin
        layout.addRow("Temperature:", temp_spin)

        conf_spin = QDoubleSpinBox()
        conf_spin.setRange(0.0, 1.0)
        conf_spin.setSingleStep(0.05)
        conf_spin.setDecimals(2)
        conf_spin.setToolTip(
            "Limiar mínimo de confiança para aceitar um segmento transcrito.\n"
            "Calculado a partir do avg_logprob retornado pelo Whisper.\n"
            "• 0.0 — desabilitado (aceita tudo)\n"
            "• 0.10–0.20 — filtra segmentos de baixíssima confiança\n"
            "• > 0.30 — pode descartar fala em ambiente ruidoso\n"
            "Recomendado: 0.0 (desabilitado) ou 0.10 como ponto de partida."
        )
        self._ui_elements["api"]["confidence_threshold"] = conf_spin
        layout.addRow("Conf. Mínima:", conf_spin)

        layout.addRow(QLabel("<hr><h2>Hugging Face (Diarização)</h2>"))

        layout.addRow(
            "Token HF:",
            self._create_password_input(
                "api", "huggingface_token",
                tooltip=(
                    "Token de acesso ao HuggingFace, necessário para baixar os modelos de diarização.\n"
                    "Obtenha em huggingface.co/settings/tokens.\n"
                    "Você também precisa aceitar os termos de uso do modelo pyannote/speaker-diarization-3.1."
                )
            )
        )

        return page

    def _build_audio_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)

        sr_cb = QComboBox()
        sr_cb.addItems(["16000", "24000", "44100", "48000"])
        sr_cb.setToolTip(
            "Taxa de amostragem em Hz usada na captura do áudio.\n"
            "• 16000 Hz — padrão para transcrição de voz (recomendado)\n"
            "• Valores maiores aumentam qualidade mas também o tamanho dos chunks enviados à API."
        )
        self._ui_elements["audio"]["sample_rate"] = sr_cb
        layout.addRow("Taxa de Amostragem:", sr_cb)

        channels_cb = QComboBox()
        channels_cb.addItems(["1", "2"])
        channels_cb.setToolTip(
            "Número de canais de áudio capturados.\n"
            "• 1 (Mono) — recomendado para transcrição de voz\n"
            "• 2 (Estéreo) — captura dois canais; o app realiza downmix para mono internamente."
        )
        self._ui_elements["audio"]["channels"] = channels_cb
        layout.addRow("Canais:", channels_cb)

        debug_save_cb = QCheckBox("Salvar chunks de áudio para Debug (.wav)")
        debug_save_cb.setToolTip(
            "Se ativado, cada fragmento de áudio detectado será salvo na pasta 'debug_audio_chunks'\n"
            "na raiz do projeto. Isso é útil para ouvir o que exatamente o Whisper está recebendo.\n"
            "Aviso: Pode encher o disco rapidamente se deixado ligado por dias."
        )
        self._ui_elements["audio"]["save_debug_audio"] = debug_save_cb
        layout.addRow("Debug:", debug_save_cb)

        aec_cb = QCheckBox("Habilitar Cancelamento Acústico de Eco (AEC)")
        aec_cb.setToolTip(
            "Quando ativado no modo 'Mic + Sistema', o aplicativo utiliza um filtro\n"
            "adaptativo (NLMS) interno via software para suprimir o som do sistema\n"
            "que vaza pelas suas caixas de som de volta ao microfone.\n\n"
            "Isso limpa significativamente os ecos e impede que vozes vindas de pessoas\n"
            "na mesma call reproduzam transcrições duplicadas, com apenas ~3s de aquecimento inicial.\n\n"
            "Não tem impacto de CPU e não depende do Windows."
        )
        self._ui_elements["audio"]["use_windows_aec"] = aec_cb
        layout.addRow("Cancelamento de Eco:", aec_cb)

        normalize_cb = QCheckBox("Normalização automática de volume do microfone")
        normalize_cb.setToolTip(
            "Ativa o AGC (Automatic Gain Control) por software para o microfone.\n\n"
            "Quando seu microfone tem volume muito baixo, essa opção amplifica\n"
            "o sinal automaticamente para manter um nível de voz consistente,\n"
            "melhorando a detecção de fala (VAD) e a qualidade da transcrição.\n\n"
            "Como funciona:\n"
            "• Calcula o nível sonoro de cada quadro de áudio (~32ms)\n"
            "• Ajusta o ganho suavemente para atingir o nível alvo\n"
            "• Não amplifica silêncio (noise gate automático)\n"
            "• Limita o ganho máximo para evitar distorção\n\n"
            "Recomendado: ativado para microfones de headset ou USB de baixo volume.\n"
            "Desative se estiver usando um microfone profissional já configurado."
        )
        self._ui_elements["audio"]["mic_normalize"] = normalize_cb
        layout.addRow("Normalização de Volume:", normalize_cb)

        return page

    def _build_vad_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)

        enabled_cb = QCheckBox("Ativar Detecção de Atividade de Voz")
        enabled_cb.setToolTip(
            "Habilita o VAD (Voice Activity Detection) usando Silero-VAD.\n"
            "Quando ativo, o áudio só é enviado para transcrição quando há fala detectada,\n"
            "economizando chamadas de API e melhorando a qualidade dos resultados."
        )
        self._ui_elements["vad"]["enabled"] = enabled_cb
        layout.addRow("", enabled_cb)

        layout.addRow(
            "Limiar de Início (Onset):",
            self._create_slider_input(
                "vad", "onset_threshold", 0.1, 0.9,
                tooltip=(
                    "Probabilidade mínima de fala para iniciar um chunk.\n"
                    "Valor mais alto → menos sensível (ignora ruídos, respirações).\n"
                    "Valor mais baixo → mais sensível (captura sons fracos).\n"
                    "Recomendado: 0.30"
                )
            )
        )

        layout.addRow(
            "Limiar de Fim (Offset):",
            self._create_slider_input(
                "vad", "offset_threshold", 0.1, 0.9,
                tooltip=(
                    "Probabilidade abaixo da qual o silêncio começa a ser contado.\n"
                    "Deve ser ≤ Onset. Valor mais baixo → o chunk encerra mais rápido após uma pausa.\n"
                    "Recomendado: 0.10"
                )
            )
        )

        onset_frames = QSpinBox()
        onset_frames.setRange(1, 10)
        onset_frames.setToolTip(
            "Número de frames consecutivos acima do limiar de início necessários para confirmar fala.\n"
            "Cada frame dura ~32ms. Valor menor → detecção mais rápida; valor maior → menos falsos positivos.\n"
            "Recomendado: 2"
        )
        self._ui_elements["vad"]["onset_frames"] = onset_frames
        layout.addRow("Frames de Confirmação (Início):", onset_frames)

        offset_frames = QSpinBox()
        offset_frames.setRange(1, 20)
        offset_frames.setToolTip(
            "Número de frames consecutivos abaixo do limiar de fim para encerrar o chunk.\n"
            "Cada frame dura ~32ms. Valor menor → chunk encerra mais rápido após pausa; valor maior → mais tolerante a pausas naturais.\n"
            "Recomendado: 20"
        )
        self._ui_elements["vad"]["offset_frames"] = offset_frames
        layout.addRow("Frames de Confirmação (Fim):", offset_frames)

        min_speech = QSpinBox()
        min_speech.setRange(50, 1000)
        min_speech.setSuffix(" ms")
        min_speech.setToolTip(
            "Duração mínima de um chunk de fala para ser enviado à transcrição.\n"
            "Chunks menores que este valor são descartados (evita enviar ruídos curtos).\n"
            "Recomendado: 500ms"
        )
        self._ui_elements["vad"]["min_speech_duration_ms"] = min_speech
        layout.addRow("Duração Mínima de Fala:", min_speech)

        speech_pad = QSpinBox()
        speech_pad.setRange(0, 1000)
        speech_pad.setSuffix(" ms")
        speech_pad.setToolTip(
            "Quantidade de áudio capturada ANTES do início detectado da fala (pre-roll).\n"
            "Evita cortar o início de palavras. Valor muito baixo causa palavras cortadas.\n"
            "Recomendado: 500ms"
        )
        self._ui_elements["vad"]["speech_pad_ms"] = speech_pad
        layout.addRow("Pre-roll de Áudio:", speech_pad)

        max_chunk = QSpinBox()
        max_chunk.setRange(5, 60)
        max_chunk.setSuffix(" s")
        max_chunk.setToolTip(
            "Duração máxima de um chunk antes de ser forçadamente encerrado e enviado.\n"
            "Evita que trechos longos de fala contínua travem a transcrição.\n"
            "Recomendado: 15s"
        )
        self._ui_elements["vad"]["max_chunk_duration_s"] = max_chunk
        layout.addRow("Corte Máximo do Chunk:", max_chunk)

        return page

    def _build_diarization_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)

        enabled_cb = QCheckBox("Ativar Identificação de Falantes")
        enabled_cb.setToolTip(
            "Habilita a diarização — identifica e rotula quem está falando em cada trecho.\n"
            "Requer token do HuggingFace e aceitação dos termos do modelo pyannote.\n"
            "Na primeira execução, baixa aproximadamente 2GB de modelos."
        )
        self._ui_elements["diarization"]["enabled"] = enabled_cb
        layout.addRow("", enabled_cb)

        info_label = QLabel(
            "<b>Atenção:</b> A primeira execução baixa ~2GB de modelos.<br>"
            "Forneça seu Token HuggingFace na aba <i>Chaves de API</i><br>"
            "e aceite os termos em <i>pyannote/speaker-diarization-3.1</i>."
        )
        info_label.setStyleSheet("color: #fabd2f;")
        layout.addRow("", info_label)

        layout.addRow(QLabel("<hr><h2>Separação e Deduplicação</h2>"))

        enable_sep_cb = QCheckBox("Habilitar Separação Proativa de Fontes")
        enable_sep_cb.setToolTip(
            "Isola fisicamente as trilhas de vozes em tempo real na VRAM para impedir overlaps e alucinações."
        )
        self._ui_elements["diarization"]["enable_source_separation"] = enable_sep_cb
        layout.addRow("", enable_sep_cb)
        
        sep_model_combo = QComboBox()
        sep_model_combo.addItems(["Conv-TasNet (Fast)", "SepFormer (Studio)"])
        sep_model_combo.setToolTip(
            "Conv-TasNet: Leve e ultrarrápido na GPU (~0.4s).\n"
            "SepFormer: Arquitetura Transformer gigante, extingue o ruído de fundo completamente, mas custa alta VRAM e lentidão (~1.8s)."
        )
        self._ui_elements["diarization"]["separator_model"] = sep_model_combo
        layout.addRow("Modelo de Separação (SOT):", sep_model_combo)

        layout.addRow(
            "Limiar do Porteiro (Overlap):",
            self._create_slider_input(
                "diarization", "overlap_threshold", 0.05, 0.40,
                tooltip=(
                    "Sensibilidade para o radar detectar Overlap de falantes.\n"
                    "• Valor alto (>0.25) → sensível, qualquer variação liga o separador.\n"
                    "• Valor baixo (<0.10) → tolerante, exige forte sobreposição pra separar.\n"
                    "Recomendado: 0.15"
                )
            )
        )

        layout.addRow(
            "Limiar de Deduplicação Levenshtein:",
            self._create_slider_input(
                "diarization", "levenshtein_threshold", 0.50, 0.99,
                tooltip=(
                    "Se dois textos na mesma janela baterem essa taxa de similaridade,\n"
                    "a cópia com menor confiança (Fantasma de separação) é deletada.\n"
                    "Recomendado: 0.85"
                )
            )
        )
        
        layout.addRow(QLabel("<hr><h2>Configuração de Embedding</h2>"))

        layout.addRow(
            "Limiar de Similaridade de Voz:",
            self._create_slider_input(
                "diarization", "similarity_threshold", 0.1, 0.95,
                tooltip=(
                    "Similaridade mínima (cosine) entre embeddings de voz para reconhecer o mesmo falante.\n"
                    "• Valor alto (>0.7) → mais restrito, cria mais falantes distintos\n"
                    "• Valor baixo (<0.5) → mais permissivo, pode confundir falantes diferentes\n"
                    "Recomendado: 0.65"
                )
            )
        )

        self._speaker_text_edit = QPlainTextEdit()
        self._speaker_text_edit.setPlaceholderText("SPEAKER_00=Leonardo\nSPEAKER_01=Ana")
        self._speaker_text_edit.setMinimumHeight(100)
        self._speaker_text_edit.setToolTip(
            "Mapeamento de nomes personalizados para os falantes detectados.\n"
            "Formato: ID_DO_FALANTE=Nome (um por linha).\n"
            "O ID é gerado automaticamente (SPEAKER_00, SPEAKER_01, etc.).\n"
            "Exemplo:\n  SPEAKER_00=Leonardo\n  SPEAKER_01=Convidado"
        )
        layout.addRow(
            QLabel("Nomes dos Falantes:<br><small>(Formato: ID=Nome, um por linha)</small>"),
            self._speaker_text_edit
        )

        return page

    def _build_server_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)

        ws_cb = QCheckBox("Ativar Servidor WebSocket")
        ws_cb.setToolTip(
            "Inicia um servidor WebSocket local que transmite a transcrição em tempo real.\n"
            "Usado pelo plugin Obsidian (WhisperPlus) e outras integrações externas."
        )
        self._ui_elements["server"]["websocket_enabled"] = ws_cb
        layout.addRow("", ws_cb)

        ws_port = QSpinBox()
        ws_port.setRange(1024, 65535)
        ws_port.setToolTip(
            "Porta TCP onde o servidor WebSocket ficará escutando.\n"
            "Padrão: 8765. Altere apenas se houver conflito com outra aplicação."
        )
        self._ui_elements["server"]["websocket_port"] = ws_port
        layout.addRow("Porta WebSocket:", ws_port)

        rest_cb = QCheckBox("Ativar Servidor REST API")
        rest_cb.setToolTip(
            "Inicia uma API REST local (FastAPI) para controlar o app e consultar transcrições via HTTP.\n"
            "Permite integração com n8n, scripts externos, Obsidian, etc."
        )
        self._ui_elements["server"]["rest_api_enabled"] = rest_cb
        layout.addRow("", rest_cb)

        rest_port = QSpinBox()
        rest_port.setRange(1024, 65535)
        rest_port.setToolTip(
            "Porta TCP onde a REST API ficará escutando.\n"
            "Padrão: 8766. Altere apenas se houver conflito com outra aplicação."
        )
        self._ui_elements["server"]["rest_api_port"] = rest_port
        layout.addRow("Porta REST API:", rest_port)

        return page

    def _build_interface_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)

        theme_cb = QComboBox()
        theme_cb.addItems(["gruvbox_dark", "light", "dark"])
        theme_cb.setToolTip(
            "Tema visual da interface.\n"
            "• gruvbox_dark — tema escuro com cores quentes (padrão)\n"
            "A mudança de tema requer reinicialização do app."
        )
        self._ui_elements["ui"]["theme"] = theme_cb
        layout.addRow("Tema:", theme_cb)

        tray_cb = QCheckBox("Minimizar para o tray ao fechar a janela")
        tray_cb.setToolTip(
            "Quando ativado, fechar a janela principal oculta o app na área de notificação (bandeja do sistema).\n"
            "O app continua rodando em segundo plano.\n"
            "Para encerrar completamente, clique com o botão direito no ícone do tray → 'Encerrar'."
        )
        self._ui_elements["ui"]["minimize_to_tray"] = tray_cb
        layout.addRow("", tray_cb)

        return page

    def _build_filters_page(self) -> QWidget:
        """Página de configuração dos filtros de alucinação do Whisper."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(QLabel("<h2>Filtros de Alucinação</h2>"))
        intro = QLabel(
            "O Whisper frequentemente transcreve ruído ou silêncio como frases específicas.\n"
            "Configure abaixo quais frases devem ser descartadas automaticamente."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        enable_rep_cb = QCheckBox("Habilitar Filtro de Repetição")
        enable_rep_cb.setToolTip(
            "Descarta segmentos que sejam iguais ou muito similares (>85%) a um segmento recente.\n"
            "Útil para eliminar alucinações em loop do Whisper.\n"
            "Desative se perceber que falas legítimas repetidas estão sendo suprimidas."
        )
        self._ui_elements["filters"]["enable_repetition"] = enable_rep_cb
        layout.addWidget(enable_rep_cb)

        layout.addSpacing(15)

        enable_pref_cb = QCheckBox("Habilitar Filtro por Prefixo")
        enable_pref_cb.setToolTip(
            "Se desativado, o aplicativo irá transcrever absolutamente tudo que tenha esse prefixo,\n"
            "incluindo os ruídos repetitivos do YouTube (ex: 'thanks for watching')."
        )
        self._ui_elements["filters"]["enable_prefixes"] = enable_pref_cb
        layout.addWidget(enable_pref_cb)

        layout.addWidget(QLabel("<b>Filtro por Prefixo</b> — descarta quando o texto <i>começa</i> com a frase:"))
        self._filter_prefixes_edit = QPlainTextEdit()
        self._filter_prefixes_edit.setToolTip(
            "Uma frase por linha.\n"
            "Segmentos cujo texto COMEÇA com qualquer uma dessas frases serão descartados.\n"
            "Ideal para padrões de dataset de treino (YouTube: 'thanks for watching', etc)."
        )
        self._filter_prefixes_edit.setMaximumHeight(140)
        layout.addWidget(self._filter_prefixes_edit)
        
        # Spacer visual
        layout.addSpacing(15)

        enable_exact_cb = QCheckBox("Habilitar Filtro Exact-Match")
        enable_exact_cb.setToolTip(
            "Se desativado, frases exatas muito curtas não serão descartadas automaticamente."
        )
        self._ui_elements["filters"]["enable_exact"] = enable_exact_cb
        layout.addWidget(enable_exact_cb)

        layout.addWidget(QLabel("<b>Filtro Exact-Match</b> — descarta quando o texto <i>é apenas</i> a frase:"))
        self._filter_exact_edit = QPlainTextEdit()
        self._filter_exact_edit.setToolTip(
            "Uma frase por linha.\n"
            "Segmentos cujo texto É EXATAMENTE essa frase serão descartados.\n"
            "Ex: 'e aí' sozinho é ruído, mas 'e aí, você viu?' é fala real.\n"
            "Use para interjeições curtas que o modelo confunde com ruído."
        )
        self._filter_exact_edit.setMaximumHeight(140)
        layout.addWidget(self._filter_exact_edit)

        layout.addStretch()
        return page

    # ── Carregar e Salvar ─────────────────────────────────────────────────────

    def _load_values(self) -> None:
        """Preenche os formulários com os valores atuais das configurações."""
        settings_dict = self._settings.to_dict()

        # Elemento especial: nomes dos falantes
        diar = settings_dict.get("diarization", {})
        labels = diar.get("speaker_labels", {})
        lines = [f"{k}={v}" for k, v in labels.items()]
        self._speaker_text_edit.setPlainText("\n".join(lines))

        # Elemento especial: filtros de alucinação
        filters = settings_dict.get("filters", {})
        self._filter_prefixes_edit.setPlainText(
            "\n".join(filters.get("hallucination_prefixes") or [])
        )
        self._filter_exact_edit.setPlainText(
            "\n".join(filters.get("hallucination_exact") or [])
        )

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
        """Coleta valores dos widgets e salva nas configurações."""
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

        # Elemento especial: nomes dos falantes
        lines = self._speaker_text_edit.toPlainText().strip().split("\n")
        speaker_map = {}
        for line in lines:
            if "=" in line:
                k, v = line.split("=", 1)
                speaker_map[k.strip()] = v.strip()
        self._settings.update_section("diarization", {"speaker_labels": speaker_map})

        # Elemento especial: filtros de alucinação (listas de strings)
        prefixes = [
            ln.strip() for ln in self._filter_prefixes_edit.toPlainText().splitlines()
            if ln.strip()
        ]
        exact = [
            ln.strip() for ln in self._filter_exact_edit.toPlainText().splitlines()
            if ln.strip()
        ]
        self._settings.update_section("filters", {
            "hallucination_prefixes": prefixes,
            "hallucination_exact": exact,
        })

        self._settings.save_now()
        self.settings_saved.emit()
        self.accept()
