"""
pipeline/memory.py
===================
Session logging (extracted from backend.py's Cell 3 logger setup,
originally lines 68-105) and the chat-history formatter (backend.py:1453-1454,
Cell 21 -- currently a no-op stub in the source, kept as-is here).

FlushFileHandler flushes to disk after every record; without it, log lines
buffer in memory and are only guaranteed to reach disk if the process exits
cleanly.
"""

import hashlib
import logging
from datetime import datetime

import config

LOG_DIR = config.LOG_DIR

_session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_log_path   = f"{LOG_DIR}/session_{_session_ts}.log"


class FlushFileHandler(logging.FileHandler):
    """Subclass that flushes to disk after every record."""
    def emit(self, record):
        super().emit(record)
        self.flush()


_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
_fh  = FlushFileHandler(_log_path, encoding="utf-8")
_ch  = logging.StreamHandler()
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)

logger = logging.getLogger("aum_chatbot")
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.addHandler(_fh)
logger.addHandler(_ch)
logger.propagate = False

logger.info("AUM-Chatbot v4.1 session started")
logger.info(f"Log file: {_log_path}")


def query_fingerprint(text: str) -> str:
    """A short, non-reversible stand-in for raw query/question text in log
    lines (Task 44, external review 2026-10-08). Several log call sites
    across pipeline/classifier.py, pipeline/semantic_router.py, and
    server/api_server.py used to embed the literal query text (sometimes
    the complete, untruncated text) to make a routing/retrieval decision
    debuggable after the fact -- but that bypassed Task 38's entire
    redaction effort, which only ever covered the /api/feedback persistence
    path, not the ordinary application logger. This still lets the same
    query be correlated across multiple log lines within one request
    (same input -> same fingerprint) without the log file itself ever
    containing what was actually asked. Routing/retrieval happens before a
    query_id is minted (query_id is assigned per-domain, after routing
    decides which domain handles the request), so this is a content
    fingerprint, not a request identifier -- callers that already have a
    query_id should log that too, not use this as a substitute for it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def format_history(history, max_turns=4, max_chars=1000):
    """
    Kept as a no-op stub, matching backend.py:1453-1454 exactly -- history
    injection into prompts was removed (see build_cos_answer_streaming's
    unused history_text parameter, kept only for signature compatibility).
    """
    return ""
