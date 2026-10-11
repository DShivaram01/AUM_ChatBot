"""Task 57 phase 2: SQLite session store (history + last quiz)."""

import os
import stat
import subprocess
import sys
import tempfile
import textwrap

from stores.session_store import SessionStore


def test_dict_like_contract_and_namespaces():
    a, b = SessionStore("t_a"), SessionStore("t_b")
    a.clear(); b.clear()
    assert "s1" not in a and a.get("s1", []) == []
    a["s1"] = {"quiz": {"questions": [1, 2]}, "count": 2}
    assert "s1" in a and a["s1"]["count"] == 2 and "s1" not in b
    a.clear()
    assert "s1" not in a


def test_expired_rows_are_invisible_and_purged():
    s = SessionStore("t_ttl", ttl_s=-1)  # everything is already expired
    s["s1"] = [1]
    assert "s1" not in s


def test_file_store_survives_restart_and_is_private():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sessions.db")
        script = textwrap.dedent("""
            import sys; sys.path.insert(0, ".")
            from stores.session_store import SessionStore
            s = SessionStore("last_quiz")
            if sys.argv[1] == "write": s["sess"] = {"topic": "guest policy"}
            else: print(s.get("sess", {}).get("topic"))
        """)
        env = {**os.environ, "AUM_SESSION_STORE_PATH": path}
        subprocess.run([sys.executable, "-c", script, "write"], env=env, check=True)
        out = subprocess.run([sys.executable, "-c", script, "read"], env=env, check=True,
                             capture_output=True, text=True).stdout.strip()
        assert out == "guest policy"
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_history_keeps_last_four_turns():
    from server import api_server
    api_server._session_history.clear()
    for i in range(6):
        api_server._record_turn("hs", f"q{i}", f"a{i}")
    assert [t["question"] for t in api_server._get_history("hs")] == ["q2", "q3", "q4", "q5"]
    assert api_server._get_history(None) == [] and api_server._get_history("other") == []
