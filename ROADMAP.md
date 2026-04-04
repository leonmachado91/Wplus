# Transcription App — Roadmap de Implementação

## Visão Geral

| Fase | Nome | Foco | Status |
|---|---|---|---|
| 1 | Foundation | Skeleton do app, settings, servidores | **Concluído** |
| 2 | Audio Pipeline | Captura → VAD → Groq → texto | **Concluído** |
| 3 | UI + Modo 2 | Interface completa, WS broadcasting, ModeController, REST | **Concluído** |
| 4 | Modo 1 | File watcher, processamento em fila, YAML frontmatter | **Concluído** |
| 5 | Diarização | Speaker labels, pyannote integração | Pendente |
| 6 | Modo 3 | Floating button, injeção de teclado | Pendente |
| 7 | Polish | Estabilidade, docs, testes de stress | Pendente |

---

## Fase 1 — Foundation

**Objetivo:** App inicia, settings funcionam, WS e REST servers aceitam conexões. Nenhum áudio ainda.

### Tarefas

- [x] **1.1** Criar estrutura de pastas completa com todos os `__init__.py`
  - Verificar: `python main.py` executa sem erro de import ✔

- [x] **1.2** Implementar `AppEvent` dataclasses (`app/core/events.py`)
  - `SessionStartedEvent`, `SessionStoppedEvent`, `SegmentAddedEvent`, `SegmentUpdatedEvent`, `ErrorEvent`
  - Verificar: dataclasses instanciam sem erro ✔

- [x] **1.3** Implementar `SettingsManager` (`app/core/settings_manager.py`)
  - Pydantic models para cada seção (ApiSettings, AudioSettings, VADSettings, etc.)
  - `load()`: lê `settings.json`, cria com defaults se não existir
  - `save()`: escreve `settings.json` (debounce 500ms)
  - `get()` / `update(section, key, value)`: acesso e mutação type-safe
  - Verificar: `settings.json` criado no primeiro run, valores persistem entre execuções ✔

- [x] **1.4** Implementar `WebSocketServer` (`app/server/websocket_server.py`)
  - `websockets.serve()` em asyncio loop em daemon thread
  - `broadcast(event_dict)`: envia JSON para todos os clientes conectados
  - Gerencia connect/disconnect sem travar
  - Thread-safety: `asyncio.run_coroutine_threadsafe` para chamar `broadcast` de fora do loop
  - Verificar: cliente ws conecta, recebe `{"event": "hello", ...}` ✔

- [x] **1.5** Implementar `RESTAPIServer` (`app/server/rest_api.py`)
  - FastAPI app com rotas: `GET /api/status`, `GET /api/settings`, `PATCH /api/settings`, `GET /api/devices`
  - uvicorn em daemon thread
  - CORS configurado
  - Verificar: `curl http://127.0.0.1:8766/api/status` retorna 200 ✔

- [x] **1.6** Implementar `ServerManager` (`app/server/server_manager.py`)
  - `start()`: inicia ambos os servidores em threads separadas
  - `stop()`: shutdown gracioso
  - Verificar: ambos os servidores sobem juntos e param sem erro ✔

- [x] **1.7** Criar `MainWindow` básico (`app/ui/main_window.py`)
  - `QMainWindow` com `QTabWidget` (abas: Live, File Watcher)
  - System tray icon com menu (Mostrar/Ocultar, Sair)
  - Carregar `gruvbox_dark.qss` na inicialização
  - Verificar: janela abre com tema Gruvbox, ícone na tray ✔

- [x] **1.8** Criar `gruvbox_dark.qss` (`app/ui/theme/gruvbox_dark.qss`)
  - Estilar: QMainWindow, QWidget, QTabBar, QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox, QScrollBar, QStatusBar, QGroupBox
  - Paleta completa Gruvbox dark ✔

- [x] **1.9** Criar `main.py`
  - `QApplication` + carrega settings + inicia servers + abre MainWindow
  - Verificar: app abre, servers respondem, fecha limpo ✔

- [x] **1.10** Testes unitários básicos
  - SettingsManager: load, save, defaults, update — testado inline ✔
  - WebSocketServer: connect, hello event, shutdown — testado inline ✔
  - REST API: /api/status, /api/settings — testado inline ✔
  - TranscriptBuffer: session, add/update segment, listeners — testado inline ✔

---

## Fase 2 — Audio Pipeline

**Objetivo:** Capturar áudio do mic → VAD → chunks → Groq → texto impresso no terminal. Sem UI ainda.

### Tarefas

- [x] **2.1** Implementar `AudioCaptureEngine` (`app/audio/capture_engine.py`)
  - `start(device_index, mode='mic'|'loopback')`: abre sounddevice InputStream (16kHz, mono, float32)
  - Callback do portaudio: `raw_pcm_queue.put_nowait(frame)` — nunca bloca
  - `stop()`: fecha stream
  - Se queue cheia: descarta frame mais antigo + log warning
  - Verificar: frames chegam na queue ao falar no mic

- [x] **2.2** Implementar `VADProcessor` (`app/audio/vad_processor.py`)
  - Carrega silero-vad uma vez ao init (torch/CPU, cache em memória)
  - Thread de processamento: `raw_pcm_queue → speech_queue`
  - State machine: SILENCE ↔ SPEECH com onset (0.5, 3 frames) e offset (0.35, 8 frames)
  - Pre-roll buffer de 300ms para não cortar início de fala
  - Emite segmento via `speech_queue` ao detectar fim de fala
  - Fallback: força emissão após 15s de fala contínua
  - Verificar: falar e parar emite segmentos; silêncio não emite nada

- [x] **2.3** Implementar `ChunkAssembler` (`app/audio/chunk_assembler.py`)
  - Recebe segmentos PCM do VADProcessor
  - Converte float32 numpy array → WAV bytes (int16, `io.BytesIO`)
  - Descarta chunks < 300ms
  - Marca `chunk_was_forced=True` para chunks do fallback de 15s
  - Coloca `(wav_bytes, chunk_meta)` na `transcription_queue`
  - Verificar: WAV bytes gerados são válidos (abrir com soundfile)

- [x] **2.4** Implementar `TranscriptSegment` dataclass (`app/transcription/segment.py`)
  - Campos: `id`, `start_time`, `end_time`, `text`, `speaker`, `confidence`, `words`, `chunk_was_forced`, `is_partial` ✔ (feito na Fase 1)

- [x] **2.5** Implementar `TranscriptionEngine` (`app/transcription/groq_engine.py`)
  - Cliente Groq async com API key das settings
  - Loop asyncio: `transcription_queue.get()` → `groq.audio.transcriptions.create()`
  - `response_format="verbose_json"`, `timestamp_granularities=["word", "segment"]`
  - Ajuste de timestamps: `session_start_offset + groq_timestamp`
  - Rate limiter: token bucket, max 18 req/min (free tier)
  - Retry: backoff exponencial 1s/2s/4s, máx 3 tentativas
  - Verificar: chunk de áudio → `TranscriptSegment` com texto correto

- [x] **2.6** Implementar `TranscriptBuffer` (`app/core/transcript_buffer.py`)
  - Lista ordenada de `TranscriptSegment` com `threading.RLock`
  - `add_segment()`, `update_segment(id, **kwargs)`, `get_all()`, `clear()`
  - `export_markdown(path)`, `export_text(path)`
  - Sistema de listeners: callbacks registrados recebem notificação em cada mutação
  - Auto-save: append ao arquivo aberto se habilitado
  - Verificar: segmentos adicionados são notificados a todos os listeners

- [x] **2.7** Adicionar WASAPI loopback ao `AudioCaptureEngine`
  - `sounddevice.WasapiSettings(loopback=True)` para captura de áudio do sistema
  - Enumerar dispositivos loopback separadamente de microfones
  - Verificar: captura áudio do YouTube/Spotify tocando

- [x] **2.8** Montar pipeline completo em modo debug
  - Conectar: Capture → VAD → Chunk → Groq → `print(segment.text)`
  - Script temporário `debug_pipeline.py`
  - Verificar: falar → texto aparece no terminal em ~6s

- [ ] **2.9** Testes unitários
  - `test_vad_processor.py`: state machine (onset, offset, fallback 15s, descarte curtos)
  - `test_chunk_assembler.py`: WAV output válido, descarte < 300ms
  - `test_transcript_buffer.py`: add, update, listeners, thread-safety, export
  - `test_groq_engine.py`: mock Groq responses, rate limiter, retry logic

---

## Fase 3 — UI + Modo 2 Completo

**Objetivo:** Interface Gruvbox funcionando, transcrição em tempo real na UI e via WebSocket.

### Tarefas

- [x] **3.1** Implementar `DeviceSelector` widget (`app/ui/device_selector.py`)
  - `QComboBox` populado com dispositivos via `sounddevice.query_devices()`
  - Botão refresh (↺)
  - Distingue mic vs loopback
  - Verificar: dropdown lista dispositivos do sistema

- [x] **3.2** Implementar `TranscriptWidget` (`app/ui/transcript_widget.py`)
  - `QTextEdit` read-only com renderização HTML
  - Cada segmento: `[HH:MM:SS]` em muted gray + speaker em cor por label + texto
  - Cores de speaker: cycling Blue→Purple→Orange→Aqua→Yellow→Green→Red
  - Indicador "processando..." animado para chunk em andamento
  - `auto_scroll` segue para o final quando habilitado
  - Menu de contexto: Copiar segmento, Copiar tudo
  - Verificar: segmentos aparecem formatados com cores corretas

- [x] **3.3** Implementar `LivePanel` (`app/ui/live_panel.py`)
  - Radio buttons: Mic / System Audio
  - `DeviceSelector` widget
  - Botões: ● REC (vermelho), ■ Stop, 💾 Export, 🗑 Clear
  - Toggle auto-save + campo de pasta + Browse
  - `TranscriptWidget` ocupando a maior parte do painel
  - Barra inferior: nível de áudio (RMS, `QProgressBar` animado) + status VAD (SPEECH/SILENCE)
  - Verificar: controles respondem a cliques, estado visual muda ao gravar

- [x] **3.4** Conectar `TranscriptBuffer` → `LivePanel` via Qt signals
  - `TranscriptBuffer` emite Qt signal (thread-safe) com `TranscriptSegment`
  - `LivePanel` recebe e atualiza `TranscriptWidget`
  - Verificar: falar → texto aparece na UI em tempo real

- [x] **3.5** Conectar `TranscriptBuffer` → `WebSocketServer`
  - Listener registrado no buffer faz `broadcast("segment_final", segment)`
  - Verificar: cliente WS conectado recebe `segment_final` com texto

- [x] **3.6** Implementar `ModeController` (`app/core/mode_controller.py`)
  - Gerencia lifecycle de cada modo
  - `start_mode("live")` → inicia Capture + VAD + Chunk + Groq pipeline
  - `stop_mode("live")` → para pipeline, salva sessão
  - Apenas um modo de captura ativo por vez (Modo 2 OU Modo 3)
  - Modo 1 pode rodar simultâneo
  - Verificar: start/stop não deixa threads orfãs

- [x] **3.7** Implementar todos os endpoints REST do Modo 2
  - `POST /api/session/start`, `POST /api/session/stop`
  - `GET /api/session/current`
  - `GET /api/transcript/current`, `GET /api/transcript/current/text`
  - `POST /api/transcript/export`, `DELETE /api/transcript/current`
  - Verificar: todos os endpoints respondem corretamente

- [x] **3.8** Implementar `SettingsDialog` (`app/ui/settings_dialog.py`)
  - Seções collapsíveis: API Keys, Audio, VAD, Diarização, Servidor, Modo 3, UI, Logs
  - Campos mascarados para API keys
  - Sliders com preview de valor para VAD thresholds
  - Salva via `SettingsManager` ao clicar OK
  - Verificar: alterar Groq API key nas settings → engine usa a nova key

- [x] **3.9** Adicionar status bar global no MainWindow
  - WS status (● verde se server ativo, ● vermelho se erro)
  - REST status
  - Contagem de clientes WS conectados
  - Verificar: UI reflete estado real dos servidores

- [x] **3.10** Sistema tray completo
  - Minimize to tray ao fechar janela (se configurado)
  - Menu: Mostrar, Gravar (toggle), Sair
  - Verificar: app continua transcrevendo com janela minimizada na tray

---

## Fase 4 — Modo 1 (File Watcher)

**Objetivo:** Dropar arquivo de áudio na pasta → arquivo .md transcrito no output.

### Tarefas

- [x] **4.1** Implementar `FileWatcherMode` (`app/modes/file_watcher_mode.py`)
  - `watchdog` `FileSystemEventHandler` para a pasta configurada
  - 500ms settle delay após `FileCreatedEvent`
  - Filtra por extensões em `mode1.supported_extensions` (.mp3, .wav, .m4a, .ogg, .flac, .webm, .mp4)
  - Fila serial de processamento (`queue.Queue` + worker thread)
  - Verificar: criar arquivo na pasta → aparece na fila ✔

- [x] **4.2** Implementar processamento de arquivo no worker
  - `AudioChunker` via pydub + `imageio-ffmpeg` (sem dependência externa de ffmpeg)
  - Carrega → converte para WAV mono 16kHz via `pydub`
  - Arquivos > 30s: divide em chunks de 25s com overlap de 2s
  - Deduplicação de overlap por matching de palavras nas bordas
  - Envia WAV chunks ao `GroqOfflineTranscriber` (síncrono, com retry)
  - Salva output `.md` com **YAML frontmatter** Obsidian-compatible:
    - `source`, `date`, `time`, `duration`, `format`, `type`
  - Verificar: `.mp3` → `.md` com transcrição completa ✔

- [x] **4.3** Implementar `FileWatcherPanel` (`app/ui/file_watcher_panel.py`)
  - Campos de pasta Watch e Output com botões Browse
  - Botões Start/Stop watching
  - Tabela de jobs: nome, barra de progresso animada, status (na fila / processando / concluído / erro)
  - Persistência de pastas via `SettingsManager`
  - Verificar: UI reflete estado real da fila ✔

- [x] **4.4** Adicionar endpoints REST do Modo 1
  - `GET /api/watcher/status` → running + job list
  - `POST /api/watcher/start` → aceita watch_folder/output_folder overrides
  - `POST /api/watcher/stop`
  - FileWatcher injetado no `app.state` do FastAPI via `MainWindow._wire_file_watcher_to_rest()`
  - Verificar: `GET /api/watcher/status` retorna `{"running":false,"jobs":[]}` ✔

- [ ] **4.5** Emitir eventos WebSocket para Modo 1
  - `file_transcription_started`, `file_transcription_complete`
  - Verificar: cliente WS recebe eventos de progresso

- [ ] **4.6** Testes de integração
  - Arquivos de teste: 10s .wav, 3min .mp3, .m4a, arquivo corrompido
  - Verificar: cada formato é processado ou falha com mensagem clara

---

## Fase 5 — Diarização

**Objetivo:** Speaker labels aparecem na transcrição, ativados nas settings.

### Tarefas

- [ ] **5.1** Implementar `DiarizationEngine` (`app/diarization/diarization_engine.py`)
  - Carregamento lazy com `torch.hub` / HuggingFace (requer HF token)
  - `submit_chunk(wav_bytes, chunk_meta)`: envia para `ThreadPoolExecutor(max_workers=1)`
  - pyannote pipeline → `[(speaker_label, start, end)]` annotations
  - Timeout: se > 20s processando, pula o chunk (evita backlog)
  - Auto-GPU se `torch.cuda.is_available()`
  - Verificar: chunk de 10s com 2 speakers → 2 labels distintos

- [ ] **5.2** Implementar `SpeakerMapper` (`app/diarization/speaker_mapper.py`)
  - Mantém mapa de embeddings → labels persistentes na sessão
  - Cosine similarity (threshold 0.75) para identificar mesmo speaker entre chunks
  - Respeita custom names do settings (`speaker_labels`)
  - Verificar: mesma pessoa em chunks diferentes recebe mesmo label

- [ ] **5.3** Integrar diarização ao pipeline
  - `ChunkAssembler` entrega cópia do WAV ao `DiarizationEngine` em paralelo
  - `TranscriptBuffer`: segmentos emitidos como `segment_partial` quando diarização ativa
  - Após diarização: `update_segment()` → WS emite `segment_updated`
  - Verificar: UI mostra texto imediatamente, speaker aparece ~3-8s depois

- [ ] **5.4** UI para diarização
  - `TranscriptWidget` renderiza speaker colors por label
  - Loading indicator no status bar durante primeira carga do modelo
  - Toast de erro se HF token inválido ou download falhar
  - Verificar: speaker labels aparecem com cores distintas

- [ ] **5.5** Settings de diarização
  - Toggle enable/disable
  - Campo HuggingFace token (mascarado)
  - Min/max speakers hint
  - Tabela de mapeamento speaker label → nome customizado
  - Verificar: ativar diarização → model começa a carregar → indicator some após loading

- [ ] **5.6** Testes
  - `test_speaker_mapper.py`: continuidade entre chunks, custom names
  - Teste manual: gravação de 2 min com 2 pessoas → speakers identificados corretamente

---

## Fase 6 — Modo 3 (Floating Button)

**Objetivo:** Botão flutuante sempre visível → clicar → falar → texto injetado no campo focado.

### Tarefas

- [ ] **6.1** Implementar `FloatingButton` widget (`app/ui/floating_button.py`)
  - `QWidget` com `Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool`
  - `WA_TranslucentBackground` + `setWindowOpacity(settings.mode3.button_opacity)`
  - Forma circular via `QRegion` mask no `resizeEvent`
  - Drag: `mousePressEvent` + `mouseMoveEvent` + `mouseReleaseEvent` (salva posição)
  - Click vs drag: < 200ms e < 5px de movimento → click
  - Animação: `QPropertyAnimation` na cor da borda (azul → vermelho pulsando)
  - Verificar: botão visível sobre outras janelas, draggable, não aparece na taskbar

- [ ] **6.2** Implementar `FloatingButtonMode` (`app/modes/floating_button_mode.py`)
  - `toggle_recording()`: start/stop pipeline de áudio
  - Pipeline próprio (não usa o mesmo buffer do Modo 2)
  - Acumula texto transcrito em buffer local
  - Ao parar: concatena todo o texto, aguarda chunks pendentes (timeout 8s)
  - Emite `text_ready(text)` signal
  - Verificar: gravar 10s de fala → `text_ready` disparado com texto completo

- [ ] **6.3** Implementar injeção de texto
  - Método padrão: `pynput.keyboard.Controller().type(transcribed_text)`
  - Fallback clipboard: `pyperclip.copy(text)` → Ctrl+V → restaura clipboard após 500ms
  - `append_newline` adiciona `\n` após o texto se configurado
  - Verificar: texto injetado em Notepad, browser, Word

- [ ] **6.4** Implementar preview bubble
  - `QFrame` frameless, always-on-top, posicionado acima do botão
  - Mostra primeiros 80 chars + "..." se maior
  - Botões [Inject] e [✗]
  - Auto-inject após `preview_bubble_duration_ms` se sem interação
  - Verificar: bubble aparece, [✗] cancela, auto-inject funciona

- [ ] **6.5** Integração com ModeController e tray
  - ModeController gerencia exclusividade (Modo 2 OU 3 ativos)
  - System tray: menu item "Ativar Modo 3" toggle
  - Settings: configuração de botão (tamanho, opacidade, inject method)
  - Verificar: Modo 2 ativo → iniciar Modo 3 para Modo 2 primeiro

- [ ] **6.6** Testes de compatibilidade de injeção
  - Notepad (pynput type)
  - Chrome address bar (pynput type)
  - VS Code (pynput type)
  - Obsidian (clipboard paste como fallback)
  - Documentar comportamento esperado por app

---

## Fase 7 — Polish & Hardening

**Objetivo:** Estabilidade production-ready, documentação, testes de stress.

### Tarefas

- [ ] **7.1** Recovery de dispositivo de áudio
  - `AudioCaptureEngine` detecta `PortAudioError` → emite `device_disconnected` signal
  - ModeController: 3 tentativas de reconexão com backoff 2s/4s/8s
  - UI: toast "Dispositivo desconectado. Reconectando..." → "Reconectado" ou "Falha"
  - Verificar: desconectar mic → app tenta reconectar → mensagem na UI

- [ ] **7.2** Tratamento de erros e mensagens ao usuário
  - Groq API key inválida → toast + status bar vermelho + instrução para ir nas settings
  - Rate limit hit → status "Aguardando rate limit..." com countdown
  - Sem conexão de rede → toast com retry
  - Nenhuma fala detectada por 60s → hint "Verifique o dispositivo de áudio"
  - Verificar: cada cenário de erro tem feedback claro na UI

- [ ] **7.3** Start with Windows (opcional)
  - Criar/remover entrada no registro `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`
  - Toggle nas settings
  - Verificar: app inicia automaticamente após restart quando habilitado

- [ ] **7.4** Logging estruturado
  - `logging.handlers.RotatingFileHandler` (10MB max, 3 backups)
  - Níveis: DEBUG (verbose pipeline), INFO (eventos principais), WARNING/ERROR (problemas)
  - Botão "Abrir log" nas settings
  - Verificar: log rotaciona corretamente, nível DEBUG não aparece em INFO

- [ ] **7.5** Performance profiling
  - CPU idle: < 5% (silero-vad não deve rodar sem fala)
  - CPU durante transcrição: < 30%
  - CPU com diarização ativa: < 60%
  - RAM base: < 500MB sem diarização, < 2GB com
  - Verificar com Task Manager e `psutil`

- [ ] **7.6** Testes de stress
  - Sessão de 1 hora contínua → sem memory leak, sem crash
  - Start/stop rápido (10x em sequência) → sem thread órfã
  - 5 clientes WS simultâneos → todos recebem eventos
  - Arquivo corrompido no Mode 1 → fila continua processando próximos

- [ ] **7.7** Documentação de integração
  - `docs/integration_guide.md`: guia completo com exemplos JS, Python, curl
  - `docs/websocket_protocol.md`: todos os eventos com schemas JSON
  - `docs/settings_reference.md`: cada campo documentado
  - Verificar: terceiro seguindo o guia consegue integrar em < 30min

- [ ] **7.8** Pinagem de dependências
  - Testar com versões especificadas no requirements.txt
  - Confirmar compatibilidade PyQt6 + torch + sounddevice no Windows 11
  - Criar `requirements-diarization.txt` separado (torch + pyannote) para instalação opcional
  - Verificar: `pip install -r requirements.txt` em ambiente limpo funciona

- [ ] **7.9** `.env.example` e onboarding
  - `.env.example` com `GROQ_API_KEY=` e `HF_TOKEN=`
  - Ao primeiro run sem Groq key: dialog de onboarding guia para inserir a key
  - Verificar: usuário novo consegue configurar e transcrever em < 5min

---

## Critérios de Conclusão por Fase

| Fase | Done When |
|---|---|
| 1 | `GET /api/status` retorna 200, WS aceita conexões, UI abre com tema Gruvbox |
| 2 | Falar no mic → texto correto aparece no terminal em < 10s |
| 3 | Falar → texto aparece na UI e no cliente WS simultaneamente |
| 4 | `.mp3` em pasta watch → `.md` transcrito em output |
| 5 | 2 speakers distintos recebem labels diferentes na transcrição |
| 6 | Clicar botão, falar, texto aparece no Notepad sem interação manual |
| 7 | 1h de sessão sem crash, guia de integração validado por terceiro |

---

## Decisões de Arquitetura Documentadas

### Por que sounddevice sobre PyAudio?
sounddevice retorna arrays NumPy diretamente (float32), tem melhor suporte a WASAPI no Windows e API mais limpa. PyAudio mantido como fallback para compatibilidade com drivers legados.

### Por que silero-vad sobre WebRTC VAD?
Silero-VAD lida melhor com ruído de fundo, música e qualidade variável de microfone. Corre em CPU com overhead mínimo (~50ms/frame). A dependência torch é compartilhada com pyannote.

### Por que chunks por pausa de fala vs tamanho fixo?
Cortar no meio de palavras degrada a qualidade do Whisper (contexto incompleto). VAD-based cutting garante que cada chunk contém frases semânticas completas. Fallback de 15s previne latência excessiva em fala contínua.

### Por que `segment_partial` + `segment_updated` para diarização?
Transcrição (Groq) é rápida (~1s). Diarização (pyannote, CPU) leva ~3-8s pelo mesmo chunk. Emitir texto imediatamente (sem speaker) e atualizar retroativamente garante que o usuário veja a transcrição sem esperar a diarização terminar.

### Por que `Qt.Tool` no Modo 3?
`Qt.Tool` é o único WindowType que previne que o clique no botão flutuante transfira o foco. Sem ele, clicar no botão mudaria o app focado e a injeção iria para o lugar errado.

### Por que websockets lib separada do FastAPI?
FastAPI WebSocket (via Starlette) adiciona overhead de routing. Para broadcast puro de eventos a N clientes simultâneos, `websockets.serve()` direto é mais simples e de menor latência.
