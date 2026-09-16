"""
server/api_server.py
======================
FastAPI wrapper around the aum_chatbot pipeline. Copied from the current
/home/gh204/Desktop/api_server.py (including the TASK 6 session-aware
routing fix) with imports updated to the new pipeline/models/ layout.

Real changes beyond import paths (see workspace.md TASK 11 log entry):
  - The old file did `import backend`, and backend.py's flat-script import
    side effect loaded every model/index. That side effect doesn't exist
    anymore -- models/loader.py and pipeline/*.py are just function
    definitions. This file's startup event now explicitly calls
    main.load_everything() (imported lazily, at startup, to avoid a
    server/ -> main.py -> server/ import-order tangle at module load
    time) to perform that load, exactly once, the same way main.py's own
    standalone entrypoint does.
  - backend.classify_topic(question) -> classifier.classify_topic(question,
    embedder, cos_index, H_index, housing_ok) -- classify_topic() no
    longer reads module globals (see pipeline/classifier.py docstring),
    so the caller must pass them. gradio_ui holds the loaded objects as
    module globals (set by init_gradio_ui()), same as backend.py did.
  - backend.cos_chat / backend.housing_chat / backend._SELECTION_RE ->
    gradio_ui.cos_chat / gradio_ui.housing_chat / gradio_ui._SELECTION_RE.
"""

import os
import sys
import time
import uuid
import asyncio
import logging
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, AsyncGenerator

# Found via a real `python server/api_server.py` launch test (TASK 12):
# Python puts the SCRIPT's own directory (server/) on sys.path[0] for a
# direct script invocation, not the project root -- so the sibling
# pipeline/, models/, config.py (and this file's own `server.gradio_ui`
# absolute import, below) aren't importable without this. Only needed for
# `python server/api_server.py`; `python -m server.api_server` run from
# the project root doesn't hit this (cwd is already on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pipeline.answer import generate_streaming
from pipeline.classifier import classify_topic, get_trace, next_query_id
import config
import server.gradio_ui as gradio_ui

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aum_api")

PIPELINE_READY = True


class ChatRequest(BaseModel):
    question: str
    topic: Literal["cos", "housing", "general", "auto"] = "auto"
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    topic_used: Literal["cos", "housing", "general_aum", "general", "open_ended_disabled", "out_of_scope"]
    query_id: str
    sources: list[str] = []
    latency_ms: int


class FeedbackRequest(BaseModel):
    reaction: Literal["up", "down"]
    scope: Literal["single", "conversation"]
    query_id: str
    session_id: str | None = None
    comment: str | None = None
    conversation: list[dict] | None = None


# ---------------------------------------------------------------------------
# ROUTING NOTE: automatic requests remain grounded: COS, Housing,
# GENERAL_AUM, or OPEN_ENDED_DISABLED. GENERAL is only selected by an explicit
# client request from the visible Open-ended mode toggle and uses the loaded
# Mistral model without AUM retrieval. GENERAL_AUM remains deterministic
# because no authoritative general-AUM collection exists.
#
# Over HTTP there is no Gradio state, so pending COS list selections are
# tracked by session. A bare-number reply to a pending list still bypasses
# general routing and goes to COS.
# ---------------------------------------------------------------------------

_session_pending: dict[str, list] = {}

_GENERAL_AUM_RESPONSE = (
    "I do not currently have an authoritative AUM source connected for that "
    "general university-information question. I can help with AUM "
    "undergraduate research symposium projects and AUM Housing and Community "
    "Standards policy."
)
_OUT_OF_SCOPE_RESPONSE = (
    "I cannot help with that request. I can help with grounded AUM research, "
    "Housing, and Community Standards questions."
)
_OPEN_ENDED_MODE_REQUIRED_RESPONSE = (
    "This chat is in grounded AUM mode. Turn on Open-ended mode to ask Mistral "
    "a general question using its pretrained knowledge."
)
_GENERAL_PROMPT = (
    "<s>[INST] Answer the user's question directly, clearly, and accurately. "
    "This is a general-purpose answer and is not grounded in the AUM source "
    "collections. Do not claim an AUM source, citation, policy, or factual "
    "basis unless the user supplied it. If you are uncertain, say so.\n\n"
    "User question: {question}\n"
    "[/INST]"
)


def _run_general_sync(question: str) -> tuple[str, str]:
    query_id = next_query_id("GP")
    prompt = _GENERAL_PROMPT.format(question=question)
    answer = ""
    for partial in generate_streaming(
        prompt, gradio_ui.llm_tok, gradio_ui.llm_model,
        query_id=query_id, max_new_tokens=350,
    ):
        answer = partial
    return answer.strip(), query_id


def _run_general_stream_sync(question: str):
    query_id = next_query_id("GP")
    prompt = _GENERAL_PROMPT.format(question=question)
    for partial in generate_streaming(
        prompt, gradio_ui.llm_tok, gradio_ui.llm_model,
        query_id=query_id, max_new_tokens=350,
    ):
        yield partial, None, "", query_id


def _non_retrieval_answer(topic: str) -> tuple[str, str] | None:
    if topic == "general_aum":
        return _GENERAL_AUM_RESPONSE, next_query_id("G")
    if topic == "out_of_scope":
        return _OUT_OF_SCOPE_RESPONSE, next_query_id("O")
    if topic == "open_ended_disabled":
        return _OPEN_ENDED_MODE_REQUIRED_RESPONSE, next_query_id("M")
    return None

# TASK 14: serializes every model.generate() call this process makes.
# Without this, two overlapping requests can both call generate() on the
# same GPU model instance at once -- observed causing a 90+ second stall
# during TASK 12 testing (workspace.md issue #9 / issues_fixes.md #9).
# Covers BOTH call sites that reach generate(): _resolve_topic() (the
# Mistral intent router added in TASK 12, when it falls through to
# classify_topic()) and _run_chat_sync/_run_chat_stream_sync (the actual
# answer). Wrapping only the latter two, as a literal reading of "lock
# around the _run_chat_sync/_run_chat_stream_sync calls" might suggest,
# would still let one request's routing call race another request's
# answer generation -- the same bug, just narrower. A single process-wide
# asyncio.Lock is correct here because this app runs as one process on one
# event loop (uvicorn.run() below, no multi-worker config) -- it would NOT
# serialize across multiple worker processes if this were ever deployed
# that way.
_generate_lock = asyncio.Lock()


def _sse_event(data: str, event: str | None = None) -> str:
    """Encode one SSE event without allowing embedded newlines to end it."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.extend(f"data: {line}" for line in str(data).splitlines() or [""])
    return "\n".join(lines) + "\n\n"


async def _resolve_topic(req: "ChatRequest") -> str:
    """
    Decides which pipeline (cos_chat/housing_chat) handles this request.
    An explicit topic always wins. For "auto": if this session has a
    pending COS selection list and the message is a bare number reply
    (matches gradio_ui._SELECTION_RE, e.g. "3"), that always routes to
    "cos".
    """
    if req.topic != "auto":
        return req.topic
    pending = _session_pending.get(req.session_id, []) if req.session_id else []
    if pending and gradio_ui._SELECTION_RE.match(req.question.strip()):
        log.info(
            f"[routing] '{req.question[:40]}' -> cos "
            f"(pending COS selection for session, overriding auto-classify)"
        )
        return "cos"
    inferred_topic = await asyncio.to_thread(
        classify_topic, req.question,
        gradio_ui.embedder, gradio_ui.cos_index, gradio_ui.H_index, gradio_ui.housing_ok,
        gradio_ui.llm_tok, gradio_ui.llm_model,
    )
    # GENERAL is an opt-in capability. The classifier may recognize a general
    # request, but only the UI's explicit topic="general" may invoke Mistral's
    # pretrained-knowledge answer path.
    if inferred_topic == "general":
        log.info("[routing] general request held in grounded mode; Open-ended mode required")
        return "open_ended_disabled"
    return inferred_topic


def _run_chat_sync(topic: str, question: str, session_id: str | None) -> tuple[str, str]:
    """Run a retrieval pipeline, GENERAL model answer, or static capability response."""
    if topic == "general":
        return _run_general_sync(question)

    static_result = _non_retrieval_answer(topic)
    if static_result is not None:
        return static_result

    pending = _session_pending.get(session_id, []) if session_id else []
    gen = (
        gradio_ui.housing_chat(question, [], [])
        if topic == "housing"
        else gradio_ui.cos_chat(question, [], pending)
    )
    final_answer = ""
    query_id = ""
    new_pending = pending
    for item in gen:
        final_answer = item[0]  # (text, pending_cands/[], debug_md, query_id)
        if item[1] is not None:
            new_pending = item[1]
        query_id = item[3]
    if session_id:
        _session_pending[session_id] = new_pending
    return final_answer, query_id


def _run_chat_stream_sync(topic: str, question: str, session_id: str | None):
    """Same as _run_chat_sync but yields partial results for the SSE endpoint."""
    if topic == "general":
        yield from _run_general_stream_sync(question)
        return

    static_result = _non_retrieval_answer(topic)
    if static_result is not None:
        answer, query_id = static_result
        yield answer, None, "", query_id
        return

    pending = _session_pending.get(session_id, []) if session_id else []
    gen = (
        gradio_ui.housing_chat(question, [], [])
        if topic == "housing"
        else gradio_ui.cos_chat(question, [], pending)
    )
    new_pending = pending
    try:
        for item in gen:
            if item[1] is not None:
                new_pending = item[1]
            yield item
    finally:
        if session_id:
            _session_pending[session_id] = new_pending


app = FastAPI(title="AUM Chatbot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_startup_lock = asyncio.Lock()
_model_loaded = False


@app.on_event("startup")
async def load_models_once() -> None:
    """
    Load embedding model, FAISS indices, reranker, and the LLM exactly once
    when the server process starts -- NOT per-request.
    """
    global _model_loaded
    async with _startup_lock:
        if _model_loaded:
            return
        # Lazy import: main.py doesn't import server/api_server.py, so
        # this doesn't create a cycle, but importing it at call time
        # (rather than at module top) keeps this file loadable on its own
        # (e.g. for py_compile / unit tests) without pulling in main.py's
        # full model-loading import chain just to define the FastAPI app.
        from main import load_everything

        log.info("Loading models and data...")
        await asyncio.to_thread(load_everything)
        assert hasattr(gradio_ui, "cos_chat") and hasattr(gradio_ui, "housing_chat")
        _model_loaded = True
        log.info("Models loaded. Server ready.")


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok" if _model_loaded else "loading",
        "pipeline_wired": PIPELINE_READY,
    }


@app.post("/api/ask", response_model=ChatResponse)
async def ask(req: ChatRequest) -> ChatResponse:
    """
    Non-streaming endpoint: good for a first working version of the client.
    Switch the client to /api/ask/stream once this works end-to-end.
    """
    if not _model_loaded:
        raise HTTPException(503, "Models still loading, try again shortly.")

    start = time.time()

    async with _generate_lock:
        topic_used = await _resolve_topic(req)
        answer, query_id = await asyncio.to_thread(_run_chat_sync, topic_used, req.question, req.session_id)

    sources: list[str] = []

    return ChatResponse(
        answer=answer,
        topic_used=topic_used,
        query_id=query_id,
        sources=sources,
        latency_ms=int((time.time() - start) * 1000),
    )


@app.post("/api/ask/stream")
async def ask_stream(req: ChatRequest) -> StreamingResponse:
    """Stream answer deltas and emit the generated query ID in the done event."""
    if not _model_loaded:
        raise HTTPException(503, "Models still loading, try again shortly.")

    async def token_stream() -> AsyncGenerator[str, None]:
        async with _generate_lock:
            topic_used = await _resolve_topic(req)

            import queue
            import threading

            q: "queue.Queue[object]" = queue.Queue()
            DONE = object()

            def producer():
                try:
                    for item in _run_chat_stream_sync(topic_used, req.question, req.session_id):
                        q.put((item[0], item[3]))
                except Exception as exc:
                    log.exception("Pipeline error during streaming")
                    q.put((f"[error] {exc}", ""))
                finally:
                    q.put(DONE)

            threading.Thread(target=producer, daemon=True).start()
            last_sent = ""
            query_id = ""
            while True:
                item = await asyncio.to_thread(q.get)
                if item is DONE:
                    break
                text, item_query_id = item
                query_id = item_query_id or query_id
                delta = text[len(last_sent):] if text.startswith(last_sent) else text
                last_sent = text
                if delta:
                    yield _sse_event(delta)

        yield _sse_event(json.dumps({'query_id': query_id}), event="done")

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest) -> dict[str, str]:
    """Append self-contained response-feedback records without losing partial reports."""
    trace_ids = [req.query_id]
    if req.scope == "conversation" and req.conversation:
        trace_ids.extend(
            message.get("query_id") for message in req.conversation
            if isinstance(message, dict) and message.get("query_id")
        )

    traces = []
    seen = set()
    for query_id in trace_ids:
        if query_id in seen:
            continue
        seen.add(query_id)
        trace = get_trace(query_id)
        if trace is not None:
            traces.append(asdict(trace))

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reaction": req.reaction,
        "scope": req.scope,
        "session_id": req.session_id,
        "comment": req.comment,
        "flagged_query_id": req.query_id,
        "traces": traces,
        # Preserve client context when traces were evicted on server restart,
        # or when the user selected conversation scope.
        "conversation": req.conversation if req.scope == "conversation" else None,
    }
    feedback_path = Path(config.LOG_DIR) / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("a", encoding="utf-8") as feedback_file:
        feedback_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import socket

    def find_free_port(preferred: int = 8000) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", preferred))
                return preferred
            except OSError:
                s.bind(("0.0.0.0", 0))
                return s.getsockname()[1]

    port = find_free_port()
    log.info(f"Starting AUM API server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
