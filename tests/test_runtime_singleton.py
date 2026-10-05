"""Phase-0 lock regression tests; these never initialize a model."""

from core.runtime_lock import RuntimeLock, RuntimeLockError


def test_runtime_lock_rejects_second_owner_and_recovers_stale_pid(tmp_path):
    path = tmp_path / "runtime.lock"
    path.write_text("999999")  # stale metadata alone never owns a flock

    owner = RuntimeLock(path)
    owner.acquire()
    assert path.read_text().strip()

    contender = RuntimeLock(path)
    try:
        contender.acquire()
    except RuntimeLockError:
        pass
    else:
        raise AssertionError("second process acquired the model-runtime lock")
    finally:
        owner.release()

    recovered = RuntimeLock(path)
    recovered.acquire()
    recovered.release()
