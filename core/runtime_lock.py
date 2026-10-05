"""Process-wide lock that protects the one allowed model runtime."""

import fcntl
import os
from pathlib import Path


class RuntimeLockError(RuntimeError):
    """Raised when another process owns the model-runtime lock."""


class RuntimeLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown"
            handle.close()
            raise RuntimeLockError(
                f"AUM model runtime is already owned by PID {owner}; reuse its healthy API."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
