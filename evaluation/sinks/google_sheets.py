"""Task 39: the remote evaluation sink.

Posts sanitized EvaluationEvents to a Google Apps Script Web App bound to
the project owner's feedback-tracking Google Sheet (an NSF-reporting pilot
setup, not a long-term telemetry database -- see workspace.md TASK 39's
design note). record() enqueues and returns immediately; one background
worker thread does the actual HTTP POST with a timeout and retry/backoff,
so a Sheets/network outage can never slow down or fail the request that
triggered it (e.g. a user's feedback submission).

The webhook URL is the only thing gating writes to that sheet -- it must
come from config.GOOGLE_SHEETS_WEBHOOK_URL (environment variable), never be
logged, and never ship inside desktop/client/ or the public repo.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict

from evaluation.schemas import EvaluationEvent
from evaluation.sinks.base import EvaluationSink
from pipeline.memory import logger

# Initial attempt + two retries. Values are deliberately short -- this is a
# background worker, not a user-facing request, but an unbounded retry loop
# could still starve the queue if the webhook is down for a long stretch.
_RETRY_DELAYS_SECONDS = (0, 2, 5)
_REQUEST_TIMEOUT_SECONDS = 10


class GoogleSheetsSink(EvaluationSink):
    def __init__(self, webhook_url: str, *, max_queue: int = 500):
        if not webhook_url:
            raise ValueError("GoogleSheetsSink requires a non-empty webhook_url")
        self.webhook_url = webhook_url
        self._queue: "queue.Queue[EvaluationEvent]" = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="google-sheets-sink"
        )
        self._worker.start()

    def record(self, event: EvaluationEvent) -> None:
        """Never blocks on the network. If the queue is full (the webhook
        has been down long enough to back up `max_queue` events), the
        newest event is dropped rather than blocking the caller or growing
        unbounded -- a backpressure policy, not an accident (Task 39 spec,
        item 5)."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped += 1
            logger.warning(
                f"[GoogleSheetsSink] queue full ({self._queue.maxsize}); dropping "
                f"event {event.event_id} (dropped so far: {self._dropped})"
            )

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            self._send_with_retry(event)

    def _send_with_retry(self, event: EvaluationEvent) -> None:
        payload = json.dumps(asdict(event)).encode("utf-8")
        last_error: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS_SECONDS, start=1):
            if delay:
                time.sleep(delay)
            try:
                request = urllib.request.Request(
                    self.webhook_url,
                    data=payload,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(
                    request, timeout=_REQUEST_TIMEOUT_SECONDS
                ) as response:
                    response.read()
                return
            except Exception as exc:  # noqa: BLE001 -- any failure here must not propagate
                last_error = exc
                logger.warning(
                    f"[GoogleSheetsSink] attempt {attempt}/{len(_RETRY_DELAYS_SECONDS)} "
                    f"failed for event {event.event_id}: {exc}"
                )
        logger.error(
            f"[GoogleSheetsSink] giving up on event {event.event_id} after "
            f"{len(_RETRY_DELAYS_SECONDS)} attempts: {last_error}"
        )
