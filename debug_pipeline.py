"""Debug Pipeline — smoke test for Phase 2 audio pipeline.

Run this to verify the full chain:
  Mic → AudioCaptureEngine → VADProcessor → ChunkAssembler → TranscriptionEngine → print()

Usage:
  python debug_pipeline.py [--device INDEX] [--mode mic|loopback]

Requires GROQ_API_KEY in settings.json or .env
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("debug_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug audio pipeline")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--mode", choices=["mic", "loopback"], default="mic")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    from app.audio.capture_engine import AudioCaptureEngine

    if args.list_devices:
        devices = AudioCaptureEngine.list_devices()
        print("\n=== Audio Input Devices ===")
        for d in devices:
            marker = " [LOOPBACK]" if d.is_loopback else ""
            print(f"  [{d.index}] {d.name}{marker} ({d.max_input_channels}ch, {d.default_samplerate}Hz)")
        return

    from app.audio.vad_processor import VADProcessor
    from app.audio.chunk_assembler import ChunkAssembler
    from app.transcription.groq_engine import TranscriptionEngine
    from app.core.settings_manager import SettingsManager
    from app.core.transcript_buffer import TranscriptBuffer

    settings = SettingsManager()
    buffer = TranscriptBuffer()
    session_id = buffer.start_session()

    api_key = settings.get("api", "groq_api_key")
    if not api_key:
        logger.error("No Groq API key! Set it in settings.json → api.groq_api_key")
        sys.exit(1)

    logger.info("Session: %s", session_id)
    logger.info("Mode: %s, Device: %s", args.mode, args.device or "default")

    # listener: print segments to terminal
    def on_buffer_event(event: str, data: dict) -> None:
        if event == "segment_final":
            seg = data["segment"]
            tc_start = seg.get("start_time", 0)
            m, s = divmod(int(tc_start), 60)
            h, m = divmod(m, 60)
            speaker = seg.get("speaker", "")
            speaker_str = f" {speaker}" if speaker else ""
            print(f"\n  [{h:02d}:{m:02d}:{s:02d}]{speaker_str}  {seg['text']}")

    buffer.add_listener(on_buffer_event)

    # build pipeline
    sample_rate = settings.get("audio", "sample_rate")
    vad_settings = settings.get("vad")

    capture = AudioCaptureEngine(sample_rate=sample_rate)
    speech_queue: queue.Queue = queue.Queue()
    transcription_queue: queue.Queue = queue.Queue()

    vad = VADProcessor(
        raw_pcm_queue=capture.raw_pcm_queue,
        speech_queue=speech_queue,
        sample_rate=sample_rate,
        onset_threshold=vad_settings["onset_threshold"],
        offset_threshold=vad_settings["offset_threshold"],
        min_speech_duration_ms=vad_settings["min_speech_duration_ms"],
        max_chunk_duration_s=vad_settings["max_chunk_duration_s"],
        speech_pad_ms=vad_settings["speech_pad_ms"],
    )

    chunk_asm = ChunkAssembler(
        speech_queue=speech_queue,
        transcription_queue=transcription_queue,
        sample_rate=sample_rate,
    )

    engine = TranscriptionEngine(settings, buffer, sample_rate=sample_rate)

    # wire chunk_assembler output → transcription engine
    import threading

    def chunk_to_engine():
        while not _stop.is_set():
            try:
                wav_bytes, meta = transcription_queue.get(timeout=0.5)
                engine.submit(wav_bytes, meta)
            except queue.Empty:
                continue

    _stop = threading.Event()

    # start everything
    print("\n" + "=" * 60)
    print("  TRANSCRIPTION APP — Debug Pipeline")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device or 'system default'}")
    print(f"  Sample rate: {sample_rate}Hz")
    print(f"  VAD onset: {vad_settings['onset_threshold']}, offset: {vad_settings['offset_threshold']}")
    print("=" * 60)
    print("  Speak into the mic. Press Ctrl+C to stop.\n")

    engine.start()
    vad.start()
    chunk_asm.start()

    bridge_thread = threading.Thread(target=chunk_to_engine, daemon=True, name="chunk-bridge")
    bridge_thread.start()

    capture.start(device_index=args.device, mode=args.mode)

    # graceful shutdown
    def shutdown(sig, frame):
        print("\n\n  Stopping...")
        _stop.set()
        capture.stop()
        chunk_asm.stop()
        vad.stop()
        time.sleep(1)  # let pending chunks drain
        engine.stop()
        info = buffer.stop_session()
        print(f"\n  Session ended: {info['segment_count']} segments, {info['duration_s']:.1f}s")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # keep alive
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
