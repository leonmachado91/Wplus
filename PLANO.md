# Transcription App — Plano Completo de Implementação

## Contexto

App desktop standalone de transcrição de áudio que funciona como **backend de transcrição local** — outros apps (webapps, plugins Obsidian, hubs de RPG) se conectam via WebSocket/REST e recebem a transcrição em tempo real.

- 3 modos de captura (foco inicial: Modo 2 — mic/áudio do sistema)
- Groq API (whisper-large-v3-turbo) para transcrição
- pyannote.audio 3.x localmente para diarização (opcional, desativada por padrão)
- PyQt6 com tema Gruvbox dark

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSCRIPTION APP (processo)                 │
│                                                                 │
│  PyQt6 UI ──► ModeController                                    │
│                    ├── Mode 1: FileWatcherMode (watchdog)       │
│                    ├── Mode 2: LiveMode (mic/loopback)          │
│                    └── Mode 3: FloatingButtonMode (pynput)      │
│                         │                                       │
│              ┌──────────▼──────────────────────┐               │
│              │       Audio Pipeline             │               │
│              │  AudioCaptureEngine (sounddevice)│               │
│              │  → VADProcessor (silero-vad)     │               │
│              │  → ChunkAssembler (pause-cut)    │               │
│              │  → TranscriptionEngine (Groq)    │               │
│              │  → DiarizationEngine (pyannote)  │               │
│              └──────────┬──────────────────────┘               │
│                         ▼                                       │
│               TranscriptBuffer (estado da sessão)               │
│                    ├── WebSocketServer :8765                    │◄── Webapp RPG
│                    ├── REST API (FastAPI) :8766                 │◄── Plugin Obsidian
│                    └── FileWriter (.md/.txt)                    │◄── Qualquer cliente
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack Técnica

| Componente | Lib | Motivo |
|---|---|---|
| UI | PyQt6 | Acesso direto a libs ML no mesmo processo |
| Tema | QSS Gruvbox dark | Custom stylesheet |
| Captura de áudio | sounddevice | WASAPI loopback nativo, API NumPy |
| VAD | silero-vad (torch/CPU) | Superior ao WebRTC VAD em qualidade |
| Transcrição | groq SDK (async) | <1s por chunk, sem GPU necessária |
| Diarização | pyannote.audio 3.x | SOTA local, desativado por padrão |
| WebSocket | websockets lib | Broadcast direto sem overhead de router |
| REST API | FastAPI + uvicorn | OpenAPI automático, Pydantic validation |
| Settings | Pydantic BaseSettings + JSON | Validação + persistência simples |
| File watcher | watchdog | ReadDirectoryChangesW nativo |
| Injeção de teclado | pynput | SendInput API do Windows |
| Conversão de áudio | pydub + ffmpeg | Suporte a todos os formatos |

---

## Estrutura de Pastas

```
transcription-app/
├── main.py
├── requirements.txt
├── settings.json                    # Criado no primeiro run
├── .env.example
│
├── app/
│   ├── core/
│   │   ├── settings_manager.py      # Pydantic models + JSON load/save
│   │   ├── transcript_buffer.py     # Estado da sessão, notifica listeners
│   │   ├── mode_controller.py       # Orquestra lifecycle dos modos
│   │   └── events.py                # AppEvent dataclasses
│   │
│   ├── audio/
│   │   ├── capture_engine.py        # sounddevice, WASAPI loopback
│   │   ├── vad_processor.py         # Silero-VAD state machine
│   │   ├── chunk_assembler.py       # Corte por pausa, fallback 15s
│   │   └── audio_utils.py           # PCM helpers
│   │
│   ├── transcription/
│   │   ├── groq_engine.py           # Groq async client, rate limit, retry
│   │   └── segment.py               # TranscriptSegment dataclass
│   │
│   ├── diarization/
│   │   ├── diarization_engine.py    # pyannote pipeline, thread pool
│   │   └── speaker_mapper.py        # Continuidade de speaker entre chunks
│   │
│   ├── server/
│   │   ├── websocket_server.py      # Broadcast server asyncio
│   │   ├── rest_api.py              # Todos os routes FastAPI
│   │   └── server_manager.py        # Start/stop ambos os servidores
│   │
│   ├── modes/
│   │   ├── mode_base.py             # AbstractMode
│   │   ├── live_mode.py             # Modo 2
│   │   ├── file_watcher_mode.py     # Modo 1
│   │   └── floating_button_mode.py  # Modo 3
│   │
│   └── ui/
│       ├── main_window.py
│       ├── live_panel.py
│       ├── file_watcher_panel.py
│       ├── settings_dialog.py
│       ├── floating_button.py
│       ├── transcript_widget.py     # QTextEdit HTML com cores por speaker
│       ├── device_selector.py
│       └── theme/
│           ├── gruvbox_dark.qss
│           └── icons/
│
└── tests/
    ├── test_vad_processor.py
    ├── test_chunk_assembler.py
    ├── test_transcript_buffer.py
    ├── test_groq_engine.py
    ├── test_websocket_server.py
    └── test_rest_api.py
```

---

## Pipeline de Áudio (Modo 2)

```
Mic/Loopback → AudioCaptureEngine (16kHz, mono, float32)
    → queue.Queue (frames de 32ms / 512 samples)
    → VADProcessor thread:
         speech_prob = silero_vad(frame)
         SILENCE → SPEECH quando prob > 0.5 por 3 frames consecutivos
         SPEECH  → SILENCE quando prob < 0.35 por 8 frames (~250ms)
         Força corte se acumular > 15s sem pausa
    → ChunkAssembler:
         Converte PCM para WAV bytes (io.BytesIO)
         Descarta chunks < 300ms
    → TranscriptionEngine (asyncio):
         groq.audio.transcriptions.create(whisper-large-v3-turbo, verbose_json)
         Ajusta timestamps: session_offset + timestamps do Groq
    → TranscriptBuffer.add_segment()
         → WebSocketServer.broadcast("segment_final")
         → LivePanel Qt signal (thread-safe)
         → FileWriter (se auto-save ativo)
    → [OPCIONAL] DiarizationEngine (thread pool):
         pyannote pipeline no mesmo WAV chunk
         Retroativamente atualiza speaker nos segmentos
         → broadcast("segment_updated")
```

**Modelo de concorrência:**
- Main thread: PyQt6 event loop
- Thread A: AudioCaptureEngine (portaudio C callback)
- Thread B: VADProcessor
- Thread C: Asyncio loop (Groq, WebSocket, REST)
- Thread D: DiarizationEngine thread pool (opcional)
- Comunicação: `queue.Queue` entre threads, `asyncio.run_coroutine_threadsafe` para asyncio

---

## Diarização (pyannote.audio)

- **Desativada por padrão** — usuário habilita nas settings + insere token HuggingFace
- Carregamento lazy na primeira ativação, com loading indicator na UI
- Roda em `ThreadPoolExecutor(max_workers=1)` para evitar OOM
- **Padrão partial/final**: WebSocket emite `segment_partial` (sem speaker) imediatamente após Groq, e `segment_updated` quando pyannote termina (~3-8s depois)
- `SpeakerMapper` mantém continuidade de labels entre chunks via cosine similarity dos embeddings
- Se GPU disponível (`torch.cuda.is_available()`), pipeline move para GPU automaticamente

---

## Modelo de Settings (JSON)

```json
{
  "api": {
    "groq_api_key": "",
    "groq_model": "whisper-large-v3-turbo",
    "groq_language": null,
    "groq_prompt": "",
    "groq_temperature": 0,
    "huggingface_token": ""
  },
  "audio": {
    "input_device_index": null,
    "loopback_device_index": null,
    "sample_rate": 16000,
    "channels": 1
  },
  "vad": {
    "enabled": true,
    "onset_threshold": 0.5,
    "offset_threshold": 0.35,
    "min_speech_duration_ms": 200,
    "max_chunk_duration_s": 15,
    "speech_pad_ms": 300
  },
  "diarization": {
    "enabled": false,
    "model": "pyannote/speaker-diarization-3.1",
    "min_speakers": null,
    "max_speakers": null,
    "speaker_labels": {}
  },
  "server": {
    "websocket_enabled": true,
    "websocket_host": "127.0.0.1",
    "websocket_port": 8765,
    "rest_api_enabled": true,
    "rest_api_port": 8766,
    "api_key_enabled": false,
    "cors_origins": ["http://localhost:3000", "app://obsidian.md"]
  },
  "mode1": {
    "watch_folder": "",
    "output_folder": "",
    "output_format": "md",
    "supported_extensions": ["wav", "mp3", "m4a", "mp4", "ogg", "flac", "webm"],
    "include_timestamps": true
  },
  "mode2": {
    "audio_source": "mic",
    "auto_save_enabled": false,
    "auto_save_folder": "",
    "show_timecodes": true,
    "show_confidence": false,
    "scroll_follow": true
  },
  "mode3": {
    "audio_source": "mic",
    "inject_method": "type",
    "button_opacity": 0.85,
    "button_size": 60,
    "append_newline": false,
    "show_preview_bubble": true,
    "preview_bubble_duration_ms": 3000
  },
  "ui": {
    "theme": "gruvbox_dark",
    "font_family": "JetBrains Mono",
    "font_size": 12,
    "minimize_to_tray": true,
    "start_minimized": false,
    "start_with_windows": false
  },
  "logging": {
    "level": "INFO",
    "log_to_file": true,
    "log_file": "logs/app.log"
  }
}
```

---

## Protocolo WebSocket

**Endereço padrão:** `ws://127.0.0.1:8765`

Com API key: `ws://127.0.0.1:8765?api_key=YOUR_KEY`

### Eventos servidor → cliente

| Evento | Quando |
|---|---|
| `hello` | Ao conectar (inclui estado atual da sessão) |
| `session_started` | Modo 2/3 começa a gravar |
| `session_stopped` | Gravação encerrada |
| `segment_partial` | Texto transcrito, speaker ainda pendente (diarização ativa) |
| `segment_final` | Texto + speaker pronto (ou diarização desativada) |
| `segment_updated` | Speaker atribuído retroativamente |
| `file_transcription_started` | Modo 1 começou a processar arquivo |
| `file_transcription_complete` | Arquivo processado, output gerado |
| `status_update` | Qualquer mudança de estado do app |
| `error` | Erros recuperáveis (GROQ_RATE_LIMIT, AUDIO_DEVICE_DISCONNECTED, etc.) |

### Exemplo `segment_final`
```json
{
  "event": "segment_final",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "segment": {
    "id": "seg-uuid-001",
    "start_time": 4.21,
    "end_time": 9.85,
    "text": "Frase transcrita aqui.",
    "speaker": "SPEAKER_00",
    "confidence": 0.97,
    "words": [{"word": "Frase", "start": 4.21, "end": 4.5, "confidence": 0.99}],
    "chunk_was_forced": false
  }
}
```

### Mensagens cliente → servidor (via WebSocket)
```json
{"command": "start_live", "audio_source": "mic"}
{"command": "stop_live"}
{"command": "ping"}
```

---

## REST API (FastAPI :8766)

```
GET    /api/status
POST   /api/session/start        {mode, audio_source, auto_save}
POST   /api/session/stop
GET    /api/session/current
GET    /api/transcript/current
GET    /api/transcript/current/text   → plain text com timecodes
POST   /api/transcript/export         {format, path?}
DELETE /api/transcript/current
GET    /api/watcher/status
POST   /api/watcher/start             {watch_folder?, output_folder?}
POST   /api/watcher/stop
POST   /api/watcher/submit            {file_path}
GET    /api/settings                  (API keys redactadas)
PATCH  /api/settings
GET    /api/devices                   → mics + loopback devices
```

CORS habilitado para localhost e `app://obsidian.md`.

---

## UI Layout (Gruvbox Dark)

**Paleta:**
```
Background dark:  #1d2021   Background:       #282828   Background soft: #32302f
Foreground:       #ebdbb2   Muted:            #a89984
Red:              #fb4934   Green:            #b8bb26   Yellow:  #fabd2f
Blue:             #83a598   Purple:           #d3869b   Orange:  #fe8019
```
Speaker colors: cycling Blue → Purple → Orange → Aqua → Yellow → Green → Red

**Main Window:**
```
[Live Transcription] [File Watcher]         [⚙ Settings]    Status: ● Ready
──────────────────────────────────────────────────────────────────────────────
Source: [● Mic] [○ System Audio]    Device: [Microphone (Realtek) ▼]  [↺]
[● REC]  [■ Stop]  [💾 Export]  [🗑 Clear]
Auto-save: [☐]  Para: [pasta...                    ]  [Browse]
──────────────────────────────────────────────────────────────────────────────
│ [00:00:04] SPEAKER_00  Texto transcrito aqui, continuando a frase           │
│                        na linha seguinte.                                   │
│                                                                             │
│ [00:00:12] SPEAKER_01  Outra pessoa falando agora.                          │
│                                                                             │
│ [00:00:22] SPEAKER_00  [▓▓▓░░░] processando...                             │
──────────────────────────────────────────────────────────────────────────────
Audio: [████████░░░░] 60%   VAD: SPEECH
WS: ws://127.0.0.1:8765 ●   REST: http://127.0.0.1:8766 ●   Clients: 2
```

**File Watcher Tab:**
```
Watch Folder:  [C:/Watch/                       ] [Browse]
Output Folder: [C:/Output/                      ] [Browse]
Format: [● .md] [○ .txt] [○ Both]
[▶ Start Watching]  [■ Stop]
────────────────────────────────────────────────
▶ meeting.mp3       [████████░░] 80%   00:45 restando
● interview.wav     Na fila
✓ standup.mp3       Concluído → standup.md
✗ corrupt.ogg       Erro: formato inválido
```

**Settings Dialog — Seções:**
1. API Keys (Groq key, HuggingFace token, model selector, language, prompt)
2. Audio (mic device, loopback device)
3. VAD (onset/offset thresholds com slider, min duration, max chunk, padding)
4. Diarização (toggle, speaker hints, mapeamento de nomes)
5. Servidor (WS port, REST port, API key, CORS origins)
6. Modo 3 (inject method, opacity, size, preview bubble)
7. UI (font, tray behavior, start with Windows)
8. Logs (level, open log file)

**Floating Button (Modo 3):**
```
┌──────────┐          ┌──────────┐
│    🎤    │  idle    │    ●     │  gravando (pulsa vermelho)
│          │          │          │
└──────────┘          └──────────┘
  borda #83a598          borda #fb4934

Preview bubble:
┌────────────────────────────────┐
│ "A frase transcrita aqui..."   │
│                [Inject]  [✗]   │
└────────────────────────────────┘
```

---

## Modo 1 — File Watcher

- `watchdog` monitora pasta via ReadDirectoryChangesW
- 500ms settle delay antes de processar arquivo recém-criado
- Fila serial de processamento (evita rate limit do Groq)
- Arquivos > 30s: divididos em chunks de 25s com overlap de 5s
- Deduplicação nas bordas dos chunks por edit distance
- pydub + ffmpeg: `.mp3 .mp4 .m4a .ogg .flac .wav .webm`

---

## Modo 3 — Floating Button

- `Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool`
- `Qt.Tool` preserva foco na janela alvo — injeção vai para o campo correto
- Injeção padrão: `pynput.keyboard.Controller().type(text)` (Unicode-safe, SendInput)
- Fallback: clipboard + Ctrl+V (para apps que não respondem ao pynput)
- Preview bubble com timeout de auto-injeção configurável
- Drag para reposicionar, posição salva em settings

---

## Componentes Críticos

| Arquivo | Por que é crítico |
|---|---|
| `app/core/transcript_buffer.py` | Coração do sistema — todos os listeners passam aqui |
| `app/audio/vad_processor.py` | Qualidade da segmentação define latência e precisão |
| `app/transcription/groq_engine.py` | Rate limiting e timestamp accounting são críticos |
| `app/server/websocket_server.py` | Thread-safety com asyncio é o ponto mais delicado |
| `app/ui/theme/gruvbox_dark.qss` | Toda a estética do app |

---

## Dependencies (requirements.txt)

```
# UI
PyQt6==6.7.1

# Audio
sounddevice==0.5.1
PyAudio==0.2.14
numpy==1.26.4
scipy==1.13.1
soundfile==0.12.1
pydub==0.25.1

# VAD
torch==2.3.1
torchaudio==2.3.1
silero-vad==5.1.2

# Transcription
groq==0.9.0

# Diarization (opcional, pesado)
pyannote.audio==3.3.2
huggingface-hub==0.23.4

# Servers
fastapi==0.111.0
uvicorn==0.30.1
websockets==12.0

# Settings
pydantic==2.7.4
pydantic-settings==2.3.3

# Mode 1
watchdog==4.0.1

# Mode 3
pynput==1.7.7
pyperclip==1.9.0

# Utils
python-dotenv==1.0.1
aiofiles==23.2.1
httpx==0.27.0
```

**Python:** 3.11 ou 3.12 (recomendado 3.12)
**ffmpeg:** instalado separadamente e no PATH

---

## Verificação End-to-End

1. `python main.py` → `GET /api/status` retorna 200
2. Cliente JS conecta no WS → recebe evento `hello`
3. Clicar Start Recording → cliente recebe `session_started`
4. Falar no mic → cliente recebe `segment_final` com texto em ~6s
5. Clicar Stop → cliente recebe `session_stopped` com contagem de segmentos
6. Ativar diarização → repetir → `segment_partial` seguido de `segment_updated` com speaker
7. Modo 1: dropar `.mp3` na pasta watch → `.md` aparece em output
8. Modo 3: clicar botão flutuante, falar, texto aparece no campo focado

---

## Guia de Integração

### JavaScript/TypeScript (Webapp, Plugin Obsidian)
```javascript
const ws = new WebSocket('ws://127.0.0.1:8765');

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.event) {
    case 'segment_final':
      appendToUI(msg.segment.text, msg.segment.speaker, msg.segment.start_time);
      break;
    case 'segment_updated':
      updateSpeaker(msg.segment_id, msg.updates.speaker);
      break;
    case 'error':
      console.warn(msg.code, msg.message);
      break;
  }
};

ws.onclose = () => setTimeout(() => reconnect(), 2000); // sempre reconectar

// Controle via REST
fetch('http://127.0.0.1:8766/api/session/start', { method: 'POST' });
```

### Plugin Obsidian
```javascript
this.addRibbonIcon('mic', 'Transcrever', () => {
  this.activeLeaf = this.app.workspace.getActiveViewOfType(MarkdownView);
  fetch('http://127.0.0.1:8766/api/session/start', { method: 'POST' });
});

// No handler de segment_final:
const line = `\n> [${formatTime(seg.start_time)}] **${seg.speaker || 'Speaker'}**: ${seg.text}`;
editor.replaceSelection(line);
```

### Python asyncio
```python
import asyncio, websockets, json

async def transcription_client(on_segment):
    async for ws in websockets.connect('ws://127.0.0.1:8765'):
        try:
            async for message in ws:
                event = json.loads(message)
                if event['event'] == 'segment_final':
                    await on_segment(event['segment'])
        except websockets.ConnectionClosed:
            continue  # reconecta automaticamente
```
