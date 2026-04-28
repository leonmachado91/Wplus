"""Transcription App — entry point."""

from __future__ import annotations

import logging
import signal
import sys
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

def _global_exception_handler(exc_type, exc_value, exc_traceback):
    """Log unhandled exceptions instead of silently crashing PyQt6."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Unhandled exception: %s", exc_value, exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = _global_exception_handler

# Isolate all Model and Cache downloads (PyTorch, HF, etc.) to the project folder
import os
import signal
from pathlib import Path
_models_dir = Path(".models").absolute()
_models_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_models_dir / "huggingface")
os.environ["TORCH_HOME"] = str(_models_dir / "torch")


def _log_gpu_status() -> None:
    """Log whether CUDA (GPU) is available for local model inference."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            logger.info("GPU disponível: %s (CUDA %s) — diarização rodará na GPU", name, cuda_ver)
        else:
            logger.warning(
                "CUDA não disponível — modelos locais rodando na CPU. "
                "Reinstale o PyTorch com suporte CUDA: "
                "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
    except Exception as exc:
        logger.warning("Não foi possível verificar status da GPU: %s", exc)


_log_gpu_status()


def _graceful_exit(sig, frame) -> None:  # noqa: ANN001
    """SIGINT handler that exits before Intel MKL's Fortran runtime fires.

    Without this, pressing Ctrl+C while pyannote/scipy is active triggers:
        forrtl: error (200): program aborting due to control-C event
    which prints a stack dump and exits with code 1.  Intercepting SIGINT here
    and calling sys.exit(0) lets Python finish its own cleanup first.
    """
    logger.info("Interrupted — shutting down.")
    logging.shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, _graceful_exit)

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
