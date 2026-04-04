"""Transcription App — entry point."""

from __future__ import annotations

import logging
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# Isolate all Model and Cache downloads (PyTorch, HF, etc.) to the project folder
import os
from pathlib import Path
_models_dir = Path(".models").absolute()
_models_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_models_dir / "huggingface")
os.environ["TORCH_HOME"] = str(_models_dir / "torch")


def main() -> None:
    from PyQt6.QtWidgets import QApplication

    from app.core.settings_manager import SettingsManager
    from app.core.transcript_buffer import TranscriptBuffer
    from app.server.server_manager import ServerManager
    from app.ui.main_window import MainWindow
    
    import pydub
    from imageio_ffmpeg import get_ffmpeg_exe
    pydub.AudioSegment.converter = get_ffmpeg_exe()

    app = QApplication(sys.argv)
    app.setApplicationName("Transcription App")
    app.setOrganizationName("WPlus")

    # allow Ctrl+C to kill the app
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # core
    settings = SettingsManager()
    buffer = TranscriptBuffer()
    
    from app.core.mode_controller import ModeController
    mode_controller = ModeController(settings, buffer)

    # servers
    server_manager = ServerManager(settings, buffer, mode_controller)
    server_manager.start()

    # UI
    window = MainWindow(settings, server_manager, buffer, mode_controller)
    window.show()

    logger.info("App started")

    exit_code = app.exec()

    # cleanup
    server_manager.stop()
    settings.save_now()
    logger.info("App exited (code %d)", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
