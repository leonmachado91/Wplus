"""Mode 1 — File Watcher: monitors a folder for audio files, transcribes them, outputs .md."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from app.audio.audio_chunker import AudioChunker
from app.transcription.groq_offline import GroqOfflineTranscriber
from app.core.settings_manager import SettingsManager

logger = logging.getLogger(__name__)

# ── Job data ────────────────────────────────────────────────────────────────

class FileJob:
    """Represents a single file transcription job."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.status: str = "queued"       # queued | processing | done | error
        self.progress: float = 0.0        # 0.0 → 1.0
        self.error_msg: str = ""
        self.output_path: Optional[Path] = None
        self.created_at = datetime.now()

    def to_dict(self) -> dict:
        return {
            "file": self.file_path.name,
            "status": self.status,
            "progress": self.progress,
            "error": self.error_msg,
            "output": str(self.output_path) if self.output_path else None,
        }


# ── Watchdog handler ────────────────────────────────────────────────────────

class _AudioFileHandler(FileSystemEventHandler):
    """Pushes newly created audio files into the processing queue."""

    def __init__(self, extensions: list[str], job_queue: queue.Queue, on_job_added: Callable):
        super().__init__()
        self._extensions = [e.lower() for e in extensions]
        self._queue = job_queue
        self._on_job_added = on_job_added

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in self._extensions:
            return

        logger.info("New audio file detected: %s", path.name)
        # settle delay — let the OS finish writing the file
        time.sleep(0.5)
        job = FileJob(path)
        self._queue.put(job)
        self._on_job_added(job)


# ── Main controller ─────────────────────────────────────────────────────────

class FileWatcherMode:
    """Watches a folder for audio files and transcribes them sequentially."""

    def __init__(self, settings: SettingsManager):
        self._settings = settings
        self._observer: Optional[Observer] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._job_queue: queue.Queue[FileJob] = queue.Queue()

        # all jobs ever seen in this session (for UI display)
        self._jobs: list[FileJob] = []
        self._jobs_lock = threading.Lock()

        # listeners for UI updates
        self._listeners: list[Callable[[FileJob], None]] = []

        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def jobs(self) -> list[FileJob]:
        with self._jobs_lock:
            return list(self._jobs)

    def add_listener(self, fn: Callable[[FileJob], None]) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[FileJob], None]) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _notify(self, job: FileJob) -> None:
        for fn in self._listeners:
            try:
                fn(job)
            except Exception:
                logger.exception("FileWatcher listener error")

    def _on_job_added(self, job: FileJob) -> None:
        with self._jobs_lock:
            self._jobs.append(job)
        self._notify(job)

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, watch_folder: str | None = None, output_folder: str | None = None) -> None:
        if self._is_running:
            logger.warning("File watcher is already running.")
            return

        cfg = self._settings.settings.mode1
        watch = Path(watch_folder or cfg.watch_folder)
        output = Path(output_folder or cfg.output_folder)

        if not watch.is_dir():
            logger.error("Watch folder does not exist: %s", watch)
            return
        if not output.exists():
            output.mkdir(parents=True, exist_ok=True)

        self._stop_event.clear()
        self._is_running = True

        # start watchdog observer
        handler = _AudioFileHandler(
            extensions=cfg.supported_extensions,
            job_queue=self._job_queue,
            on_job_added=self._on_job_added,
        )
        self._observer = Observer()
        self._observer.schedule(handler, str(watch), recursive=False)
        self._observer.start()

        # start worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(output,),
            daemon=True,
            name="filewatcher-worker",
        )
        self._worker_thread.start()

        logger.info("FileWatcher started: watching=%s output=%s", watch, output)

    def stop(self) -> None:
        if not self._is_running:
            return

        self._stop_event.set()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=3)
            self._observer = None

        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None

        self._is_running = False
        logger.info("FileWatcher stopped.")

    # ── worker ───────────────────────────────────────────────────────────

    def _worker_loop(self, output_folder: Path) -> None:
        chunker = AudioChunker()
        transcriber = GroqOfflineTranscriber(self._settings)

        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._process_job(job, chunker, transcriber, output_folder)

    def _process_job(
        self,
        job: FileJob,
        chunker: AudioChunker,
        transcriber: GroqOfflineTranscriber,
        output_folder: Path,
    ) -> None:
        job.status = "processing"
        job.progress = 0.0
        self._notify(job)

        try:
            # collect all chunks first to know total
            chunks = list(chunker.slice_file(str(job.file_path)))
            if not chunks:
                job.status = "error"
                job.error_msg = "No audio data could be loaded"
                self._notify(job)
                return

            total = len(chunks)
            texts: list[str] = []

            for i, (wav_bytes, duration_ms, offset_ms) in enumerate(chunks):
                if self._stop_event.is_set():
                    job.status = "error"
                    job.error_msg = "Cancelled"
                    self._notify(job)
                    return

                text = transcriber.transcribe_chunk(wav_bytes)
                if text:
                    texts.append(text)

                job.progress = (i + 1) / total
                self._notify(job)

            # deduplicate overlap borders
            merged = self._deduplicate_overlap(texts)

            # compute total duration from last chunk
            last_offset = chunks[-1][2]
            last_duration = chunks[-1][1]
            total_duration_ms = last_offset + last_duration

            # write output .md with YAML frontmatter
            output_path = self._write_output(
                job.file_path, merged, total_duration_ms, output_folder
            )

            job.output_path = output_path
            job.status = "done"
            job.progress = 1.0
            self._notify(job)
            logger.info("Job complete: %s → %s", job.file_path.name, output_path.name)

        except Exception as e:
            logger.exception("Job failed: %s", job.file_path.name)
            job.status = "error"
            job.error_msg = str(e)
            self._notify(job)

    # ── deduplication ────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_overlap(texts: list[str]) -> str:
        """Merge consecutive texts, removing repeated words at the boundaries."""
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]

        merged = texts[0]
        for i in range(1, len(texts)):
            current = texts[i]
            # try to find overlap at worst of length 30 words
            prev_words = merged.split()
            curr_words = current.split()

            best_overlap = 0
            max_check = min(30, len(prev_words), len(curr_words))

            for length in range(1, max_check + 1):
                tail = prev_words[-length:]
                head = curr_words[:length]
                if tail == head:
                    best_overlap = length

            if best_overlap > 0:
                merged += " " + " ".join(curr_words[best_overlap:])
            else:
                merged += " " + current

        return merged.strip()

    # ── output writer ────────────────────────────────────────────────────

    @staticmethod
    def _write_output(
        source_path: Path,
        text: str,
        total_duration_ms: float,
        output_folder: Path,
    ) -> Path:
        now = datetime.now()
        stem = source_path.stem
        output_name = f"{stem}.md"
        output_path = output_folder / output_name

        # avoid overwriting
        counter = 1
        while output_path.exists():
            output_name = f"{stem}_{counter}.md"
            output_path = output_folder / output_name
            counter += 1

        duration_s = total_duration_ms / 1000.0
        minutes = int(duration_s // 60)
        seconds = int(duration_s % 60)

        frontmatter = (
            "---\n"
            f"source: \"{source_path.name}\"\n"
            f"date: {now.strftime('%Y-%m-%d')}\n"
            f"time: {now.strftime('%H:%M:%S')}\n"
            f"duration: \"{minutes}m {seconds}s\"\n"
            f"format: {source_path.suffix.lstrip('.')}\n"
            f"type: transcription\n"
            "---\n\n"
        )

        body = f"# {stem}\n\n{text}\n"

        output_path.write_text(frontmatter + body, encoding="utf-8")
        logger.info("Wrote transcription: %s", output_path)
        return output_path
