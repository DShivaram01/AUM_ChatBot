"""Task 33: FastAPI must depend on AssistantService, never Gradio state."""

import ast
from pathlib import Path


API = Path(__file__).parents[1] / "server" / "api_server.py"


def test_api_has_no_gradio_import_or_runtime_globals():
    tree = ast.parse(API.read_text(encoding="utf-8"))
    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if hasattr(node, "names") else [])
    }
    assert "server.gradio_ui" not in imported
    source = API.read_text(encoding="utf-8")
    assert "get_assistant_service" in source


def test_gradio_is_only_a_service_client():
    source = (Path(__file__).parents[1] / "server" / "gradio_ui.py").read_text(encoding="utf-8")
    assert "get_assistant_service" in source
    assert "retrieve_cos_rrf" not in source
    assert "build_cos_answer_streaming" not in source
