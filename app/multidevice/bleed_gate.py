"""Cross-device energy gate for presencial sessions (P1).

BleedGateCoordinator sits between per-participant pipelines and the transcription
engine. It collects speech chunks from all participants, groups them by temporal
overlap, and discards the non-dominant speaker in each group.

Algorithm (dual-criteria — RMS + TDOA):
  1. Each ParticipantPipeline submits finished speech chunks here instead of
     going directly to the TranscriptionEngine.
  2. A background flush thread runs every 50 ms.
  3. Chunks whose deadline has expired are grouped by temporal overlap
     (chunks with |started_at_ms - reference| < window_ms).
  4. Within a group:
       - Remote participants always pass (no bleed between remote devices).
       - If only one presencial chunk → passes directly.
       - If multiple presencial chunks → dual-criteria decision:

         [TDOA criterion — primary]
         Compare client_timestamp_ms values. The chunk with the smallest
         timestamp captured the sound first → its microphone was physically
         closer to the speaker. If the spread between timestamps is >=
         tdoa_min_ms, TDOA is considered a valid discriminator and the
         winner is the earliest chunk.  Losers are discarded regardless
         of their RMS.

         [RMS criterion — fallback]
         If the TDOA spread < tdoa_min_ms (devices at similar distance,
         or clocks too close to discriminate), fall back to the original
         energy gate: keep chunks whose RMS is within margin_db of the
         loudest one. Multiple chunks above the threshold all pass.

         When tdoa_enabled=False, only RMS criterion is used.

  5. Approved chunks are forwarded via the on_approved callback.

Design decisions (confirmed by user):
  - window_ms = 250  (lower latency; can raise if bleed persists)
  - tdoa_min_ms = 20 (minimum ms spread for TDOA to be trusted)
  - tdoa_enabled = True by default when gate is enabled
  - auto_detect = False  (user selects presencial/remoto manually)
  - enabled = False by default (gate is off for remote sessions)
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PendingChunk:
    """A finished speech chunk waiting for bleed-gate evaluation."""

    token: str
    speaker_name: str
    mode: str                   # "presencial" | "remoto" | "auto"
    started_at_ms: float        # server-normalised audio start time (for timeline ordering)
    submit_time_ms: float       # wall-clock ms when submitted to the gate (for grouping)
    deadline_ms: float          # wall-clock ms after which this chunk is flushed
    rms_mean: float
    wav_bytes: bytes
    meta: dict = field(default_factory=dict)
    # Client-side capture timestamp (ms, uint32 from frame header).
    # Reflects the physical moment the microphone captured the audio — not
    # network delivery time.  Used for TDOA-based speaker discrimination.
    client_timestamp_ms: float = 0.0


# Callback type: receives an approved PendingChunk and forwards it to Groq.
ApprovedCallback = Callable[[PendingChunk], None]


class BleedGateCoordinator:
    """Central coordinator that decides which chunks survive the bleed gate.

    One instance is shared across all ParticipantPipelines in a session and
    managed by ParticipantManager (start / stop with the session lifecycle).
    """

    def __init__(
        self,
        window_ms: float = 250.0,
        margin_db: float = 6.0,
        enabled: bool = False,
        tdoa_min_ms: float = 20.0,
        tdoa_enabled: bool = True,
    ) -> None:
        self.window_ms = window_ms
        self.enabled = enabled
        self.tdoa_min_ms = tdoa_min_ms
        self.tdoa_enabled = tdoa_enabled

        # dB → linear amplitude ratio  (e.g. 6 dB → ~2×)
        self._margin_db = margin_db
        self._margin_linear: float = 10 ** (margin_db / 20.0)

        self._lock = threading.Lock()
        self._pending: List[PendingChunk] = []
        self._on_approved: Optional[ApprovedCallback] = None

        self._stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None

    # ── public API ────────────────────────────────────────────────────────

    def set_on_approved(self, callback: ApprovedCallback) -> None:
        """Register the callback that receives approved chunks."""
        self._on_approved = callback

    def update_params(
        self,
        *,
        window_ms: Optional[float] = None,
        margin_db: Optional[float] = None,
        tdoa_min_ms: Optional[float] = None,
        tdoa_enabled: Optional[bool] = None,
    ) -> None:
        """Update gate parameters at runtime (called from UI controls)."""
        if window_ms is not None:
            self.window_ms = float(window_ms)
        if margin_db is not None:
            self._margin_db = float(margin_db)
            self._margin_linear = 10 ** (float(margin_db) / 20.0)
        if tdoa_min_ms is not None:
            self.tdoa_min_ms = float(tdoa_min_ms)
        if tdoa_enabled is not None:
            self.tdoa_enabled = bool(tdoa_enabled)
        logger.info(
            "BleedGate params updated: window=%.0fms, margin=%.1fdB, "
            "tdoa_min=%.0fms, tdoa_enabled=%s",
            self.window_ms, self._margin_db, self.tdoa_min_ms, self.tdoa_enabled,
        )

    def start(self) -> None:
        """Start the background flush thread. Call when the session starts."""
        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="bleed-gate-flush",
        )
        self._flush_thread.start()
        logger.info(
            "BleedGateCoordinator started (enabled=%s, window_ms=%.0f, "
            "tdoa_enabled=%s, tdoa_min_ms=%.0f)",
            self.enabled, self.window_ms, self.tdoa_enabled, self.tdoa_min_ms,
        )

    def stop(self) -> None:
        """Stop the flush thread. Flush remaining pending chunks before exit."""
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=2)
        # Drain anything still pending — approve everything on teardown
        with self._lock:
            remaining = list(self._pending)
            self._pending.clear()
        for chunk in remaining:
            self._approve(chunk)
        logger.info("BleedGateCoordinator stopped")

    def submit(self, chunk: PendingChunk) -> None:
        """Accept a finished speech chunk for gate evaluation.

        If the gate is disabled, the chunk is approved immediately (zero latency).
        Otherwise it enters the pending buffer and waits for the flush cycle.
        """
        if not self.enabled:
            self._approve(chunk)
            return

        with self._lock:
            self._pending.append(chunk)

    # ── flush loop ────────────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Background thread: every 50 ms check for expired chunks and process them."""
        while not self._stop_event.is_set():
            time.sleep(0.05)  # 50 ms resolution
            self._flush_expired()

    def _flush_expired(self) -> None:
        """Move expired chunks and their temporal neighbors into processing groups.

        Grouping is done by ``submit_time_ms`` (wall-clock server time), NOT by
        ``started_at_ms`` (VAD-internal time).  Each device's VAD runs its own
        independent frame counter starting from zero when that participant joins;
        devices that joined at different offsets will have wildly different
        ``started_at_ms`` values for the same physical audio moment, making
        ``started_at_ms``-based grouping unreliable.

        Using the server wall-clock at submission time means that bleed chunks
        from different devices, all triggered by the same utterance, will arrive
        within a short LAN round-trip window and will be grouped together.
        """
        now_ms = time.monotonic() * 1000.0

        with self._lock:
            if not self._pending:
                return

            # Sort by server submission time
            self._pending.sort(key=lambda c: c.submit_time_ms)

            # Check if the earliest chunk is ready to be flushed
            if now_ms < self._pending[0].deadline_ms:
                return

            expired_groups: List[List[PendingChunk]] = []

            while self._pending and now_ms >= self._pending[0].deadline_ms:
                group = [self._pending.pop(0)]
                ref_ms = group[0].submit_time_ms

                # Pull neighbors submitted within window_ms of the first chunk
                while (
                    self._pending
                    and abs(self._pending[0].submit_time_ms - ref_ms) <= self.window_ms
                ):
                    group.append(self._pending.pop(0))

                expired_groups.append(group)

        for group in expired_groups:
            self._process_group(group)

    def _process_group(self, group: List[PendingChunk]) -> None:
        """Apply the dual-criteria gate (TDOA + RMS) to one temporal group of chunks."""
        remotes = [c for c in group if c.mode == "remoto"]
        presenciais = [c for c in group if c.mode != "remoto"]

        # Remote chunks always pass — no bleed between separate rooms
        for chunk in remotes:
            self._approve(chunk)

        if not presenciais:
            return

        if len(presenciais) == 1:
            self._approve(presenciais[0])
            return

        # ── Multiple presencial chunks: run dual-criteria decision ────────

        rms_info = ", ".join(
            f"{c.speaker_name}: rms={c.rms_mean:.4f} ts={c.client_timestamp_ms:.0f}ms"
            for c in presenciais
        )

        # ── Criterion 1: TDOA (primary) ───────────────────────────────────
        if self.tdoa_enabled and all(c.client_timestamp_ms > 0 for c in presenciais):
            ts_values = [c.client_timestamp_ms for c in presenciais]
            # Handle uint32 wrap-around (~49-day cycle): if spread > 2^31 it wrapped
            ts_min = min(ts_values)
            ts_max = max(ts_values)
            delta_ms = ts_max - ts_min
            # Correct for uint32 overflow (unlikely in practice but safe)
            if delta_ms > 2_147_483_648:
                delta_ms = 4_294_967_296 - delta_ms

            if delta_ms > 150:
                logger.warning(
                    "BleedGate [TDOA] delta=%.0fms absurdo (>150ms). "
                    "Isso é dessincronia de relógio entre os celulares, não distância física. "
                    "Ignorando TDOA e usando RMS.", delta_ms
                )
            elif delta_ms >= self.tdoa_min_ms:
                # TDOA is discriminating: earliest timestamp wins
                winner = min(presenciais, key=lambda c: c.client_timestamp_ms)
                logger.info(
                    "BleedGate [TDOA] window=%.0fms delta=%.0fms | %s | "
                    "WINNER=%s — losers discarded",
                    self.window_ms, delta_ms, rms_info, winner.speaker_name,
                )
                for chunk in presenciais:
                    if chunk is winner:
                        self._approve(chunk)
                    else:
                        logger.info(
                            "BleedGate: DISCARDED %s (TDOA loser, delta=%.0fms)",
                            chunk.speaker_name, delta_ms,
                        )
                return  # TDOA handled the group — skip RMS fallback

            logger.info(
                "BleedGate [TDOA] delta=%.0fms < min=%.0fms — falling back to RMS | %s",
                delta_ms, self.tdoa_min_ms, rms_info,
            )
        elif self.tdoa_enabled:
            logger.info(
                "BleedGate [TDOA] skipped (some timestamps are 0) — falling back to RMS | %s",
                rms_info,
            )

        # ── Criterion 2: RMS (fallback) ───────────────────────────────────
        max_rms = max(c.rms_mean for c in presenciais)
        threshold_rms = max_rms / self._margin_linear

        logger.info(
            "BleedGate [RMS] window=%.0fms max=%.4f margin=%.1fdB | %s",
            self.window_ms, max_rms, self._margin_db, rms_info,
        )

        approved_count = 0
        for chunk in presenciais:
            if chunk.rms_mean >= threshold_rms:
                self._approve(chunk)
                approved_count += 1
            else:
                logger.info(
                    "BleedGate: DISCARDED %s (RMS fallback: %.4f < %.4f)",
                    chunk.speaker_name, chunk.rms_mean, threshold_rms,
                )

    # ── internal ─────────────────────────────────────────────────────────

    def _approve(self, chunk: PendingChunk) -> None:
        """Forward an approved chunk to the transcription engine via callback."""
        if self._on_approved is not None:
            try:
                self._on_approved(chunk)
            except Exception:
                logger.exception(
                    "BleedGate: on_approved callback raised for %s", chunk.speaker_name
                )
