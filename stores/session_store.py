"""Per-session JSON state (chat history, last quiz) on SQLite (Task 57 phase 2).

In-memory by default -- the same lifetime as the plain dicts this replaced.
Set AUM_SESSION_STORE_PATH to a file to make sessions survive a backend
restart (e.g. "explain question 2" after a redeploy). That file then holds
user questions, answers and document-derived quiz text, so it is created
0600 and rows expire after AUM_SESSION_TTL_HOURS (default 24, the same
lifetime as uploaded documents in DocumentService).
"""

import json
import os
import sqlite3
import threading
import time

_DB_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None


def _connection() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        path = os.environ.get("AUM_SESSION_STORE_PATH", "").strip() or ":memory:"
        if path != ":memory:" and not os.path.exists(path):
            os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        _CONN = sqlite3.connect(path, check_same_thread=False)
        _CONN.execute(
            "CREATE TABLE IF NOT EXISTS session_state ("
            " namespace TEXT, session_id TEXT, value TEXT, updated REAL,"
            " PRIMARY KEY (namespace, session_id))"
        )
        _CONN.commit()
    return _CONN


class SessionStore:
    """Dict-like view of one namespace: `sid in s`, `s[sid]`, `s[sid] = v`,
    `s.get(sid, default)`, `s.clear()`. Values must be JSON-serialisable."""

    def __init__(self, namespace: str, ttl_s: float | None = None):
        self.namespace = namespace
        self.ttl_s = ttl_s if ttl_s is not None else float(os.environ.get("AUM_SESSION_TTL_HOURS", "24")) * 3600

    def _fresh_after(self) -> float:
        return time.time() - self.ttl_s

    def get(self, session_id, default=None):
        with _DB_LOCK:
            row = _connection().execute(
                "SELECT value FROM session_state WHERE namespace=? AND session_id=? AND updated>=?",
                (self.namespace, session_id, self._fresh_after()),
            ).fetchone()
        return json.loads(row[0]) if row else default

    def __contains__(self, session_id) -> bool:
        return self.get(session_id) is not None

    def __getitem__(self, session_id):
        value = self.get(session_id)
        if value is None:
            raise KeyError(session_id)
        return value

    def __setitem__(self, session_id, value) -> None:
        with _DB_LOCK:
            conn = _connection()
            conn.execute("DELETE FROM session_state WHERE updated<?", (self._fresh_after(),))
            conn.execute(
                "INSERT OR REPLACE INTO session_state VALUES (?, ?, ?, ?)",
                (self.namespace, session_id, json.dumps(value), time.time()),
            )
            conn.commit()

    def clear(self) -> None:
        with _DB_LOCK:
            _connection().execute("DELETE FROM session_state WHERE namespace=?", (self.namespace,))
            _connection().commit()
