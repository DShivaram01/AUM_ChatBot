"""Task 38: the default sink for local development and tests. This is what
production also uses until a remote sink (Task 39) is added -- it's the
same mechanism, just sanitized and bounded instead of a raw asdict(trace)
dump, and append-only JSONL exactly like the current logs/feedback.jsonl."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from evaluation.schemas import EvaluationEvent
from evaluation.sinks.base import EvaluationSink


class LocalDevSink(EvaluationSink):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, event: EvaluationEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
