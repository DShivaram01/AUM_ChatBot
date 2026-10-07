"""Opt-in smoke test for an already-running real model runtime."""

import os
from urllib.request import urlopen

import pytest


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("AUM_RUN_GPU_TESTS") != "1",
    reason="set AUM_RUN_GPU_TESTS=1 to target an already-running real runtime",
)
def test_real_runtime_health_is_ready():
    with urlopen("http://127.0.0.1:8000/api/health", timeout=5) as response:
        assert "\"status\":\"ok\"" in response.read().decode()
