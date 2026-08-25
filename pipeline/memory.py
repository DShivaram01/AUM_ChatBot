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


def format_history(history, max_turns=4, max_chars=1000):
    """
    Kept as a no-op stub, matching backend.py:1453-1454 exactly -- history
    injection into prompts was removed (see build_cos_answer_streaming's
    unused history_text parameter, kept only for signature compatibility).
    """
    return ""
