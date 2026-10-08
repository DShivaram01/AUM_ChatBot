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
    gradio_ui.cos_chat / gradio_ui.housing_chat / get_assistant_service().selection_re.
"""

import os
import sys
import time
import uuid
import asyncio
import logging
import json
from datetime import datetime, timezone
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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pipeline.classifier import (
    QueryTrace,
    get_trace,
    next_query_id,
    store_trace,
)
from pipeline.memory import query_fingerprint
from core.runtime_lock import RuntimeLock, RuntimeLockError
import config
from core.orchestrator import get_assistant_service
from evaluation.schemas import EvaluationEvent
from evaluation.sanitizer import session_hash
from evaluation.service import EvaluationService
from evaluation.sinks.local_dev import LocalDevSink

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aum_api")

PIPELINE_READY = True

# Task 38 (external review, workspace.md Entry 055): LocalDevSink is the
# production sink too until Task 39 adds a remote one -- same mechanism,
# sanitized and bounded instead of a raw asdict(QueryTrace) dump. Swapping
# in a remote sink later means changing this constructor, not any call site.
_evaluation_service = EvaluationService(
    sinks=[LocalDevSink(Path(config.LOG_DIR) / "evaluation_events.jsonl")]
)


class ChatRequest(BaseModel):
    question: str
    topic: Literal["cos", "housing", "general", "auto"] = "auto"
    session_id: str | None = None
    document_ids: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    topic_used: Literal[
        "cos", "housing", "general_aum", "general",
        "open_ended_disabled", "out_of_scope", "document",
    ]
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


class QuizRequest(BaseModel):
    topic: str
    count: int = 10
    source_mode: Literal["pretrained", "housing", "document"] = "pretrained"
    document_ids: list[str] | None = None
    session_id: str | None = None


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
    answer = ""
    query_id = ""
    for partial, _, _, query_id in get_assistant_service().general_chat(question):
        answer = partial
    return answer.strip(), query_id


def _run_general_stream_sync(question: str):
    yield from get_assistant_service().general_chat(question)


def _non_retrieval_answer(
    topic: str, question: str, routing_path: str,
) -> tuple[str, str] | None:
    responses = {
        "general_aum": (_GENERAL_AUM_RESPONSE, "G"),
        "out_of_scope": (_OUT_OF_SCOPE_RESPONSE, "O"),
        "open_ended_disabled": (_OPEN_ENDED_MODE_REQUIRED_RESPONSE, "M"),
    }
    response = responses.get(topic)
    if response is None:
        return None

    answer, prefix = response
    query_id = next_query_id(prefix)
    store_trace(QueryTrace(
        query_id=query_id,
        query=question,
        tab="non_retrieval",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        intent_type=topic,
        routing_path=routing_path,
        response_state="non_retrieval",
        final_answer=answer,
    ))
    return answer, query_id

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


async def _resolve_topic(req: "ChatRequest") -> tuple[str, str]:
    """
    Decides which pipeline (cos_chat/housing_chat) handles this request.
    An attached document always wins (explicit document-grounded request).
    Otherwise an explicit topic always wins. For "auto": if this session has
    a pending COS selection list and the message is a bare number reply
    (matches get_assistant_service().selection_re, e.g. "3"), that always routes to
    "cos".
    """
    if req.document_ids:
        return "document", "explicit document selection"
    if req.topic != "auto":
        return req.topic, "explicit client topic"
    pending = _session_pending.get(req.session_id, []) if req.session_id else []
    if pending and get_assistant_service().selection_re.match(req.question.strip()):
        log.info(
            f"[routing] fp={query_fingerprint(req.question)} -> cos "
            f"(pending COS selection for session, overriding auto-classify)"
        )
        return "cos", "pending COS selection"
    routing_steps: list[str] = []
    inferred_topic = await asyncio.to_thread(
        get_assistant_service().classify_topic, req.question, routing_steps,
    )
    routing_path = " -> ".join(routing_steps) or "classifier route unavailable"
    # GENERAL is an opt-in capability. The classifier may recognize a general
    # request, but only the UI's explicit topic="general" may invoke Mistral's
    # pretrained-knowledge answer path.
    if inferred_topic == "general":
        log.info("[routing] general request held in grounded mode; Open-ended mode required")
        return "open_ended_disabled", routing_path + "; open-ended mode disabled"
    return inferred_topic, routing_path


def _run_chat_sync(
    topic: str, question: str, session_id: str | None, routing_path: str = "",
    document_ids: list[str] | None = None,
) -> tuple[str, str]:
    """Run a retrieval pipeline, GENERAL model answer, or static capability response."""
    if topic == "general":
        return _run_general_sync(question)

    static_result = _non_retrieval_answer(topic, question, routing_path)
    if static_result is not None:
        return static_result

    pending = _session_pending.get(session_id, []) if session_id else []
    gen = (
        get_assistant_service().document_chat(question, document_ids or [], session_id)
        if topic == "document"
        else get_assistant_service().housing_chat(question, [], [])
        if topic == "housing"
        else get_assistant_service().cos_chat(question, [], pending)
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


def _run_chat_stream_sync(
    topic: str, question: str, session_id: str | None, routing_path: str = "",
    document_ids: list[str] | None = None,
):
    """Same as _run_chat_sync but yields partial results for the SSE endpoint."""
    if topic == "general":
        yield from _run_general_stream_sync(question)
        return

    static_result = _non_retrieval_answer(topic, question, routing_path)
    if static_result is not None:
        answer, query_id = static_result
        yield answer, None, "", query_id
        return

    pending = _session_pending.get(session_id, []) if session_id else []
    gen = (
        get_assistant_service().document_chat(question, document_ids or [], session_id)
        if topic == "document"
        else get_assistant_service().housing_chat(question, [], [])
        if topic == "housing"
        else get_assistant_service().cos_chat(question, [], pending)
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
_runtime_lock: RuntimeLock | None = None


@app.on_event("startup")
async def load_models_once() -> None:
    """
    Load embedding model, FAISS indices, reranker, and the LLM exactly once
    when the server process starts -- NOT per-request.
    """
    global _model_loaded, _runtime_lock
    async with _startup_lock:
        if _model_loaded:
            return
        if os.environ.get("AUM_LOAD_LLM", "1") != "1":
            log.info("AUM_LOAD_LLM is disabled; skipping model initialization")
            return
        lock = RuntimeLock(Path(config.BASE_DIR) / "aum_chatbot_runtime.lock")
        try:
            lock.acquire()
        except RuntimeLockError:
            # Do not start a second model copy if a different API process owns
            # the runtime. The configured fixed port will direct clients to it.
            raise
        # Lazy import: main.py doesn't import server/api_server.py, so
        # this doesn't create a cycle, but importing it at call time
        # (rather than at module top) keeps this file loadable on its own
        # (e.g. for py_compile / unit tests) without pulling in main.py's
        # full model-loading import chain just to define the FastAPI app.
        from main import load_everything

        try:
            log.info("Loading models and data...")
            await asyncio.to_thread(load_everything)
            assert get_assistant_service().runtime.ready
            _model_loaded = True
            _runtime_lock = lock
            log.info("Models loaded. Server ready.")
        except Exception:
            lock.release()
            raise


@app.on_event("shutdown")
async def release_runtime_lock() -> None:
    global _runtime_lock
    if _runtime_lock is not None:
        _runtime_lock.release()
        _runtime_lock = None


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok" if _model_loaded else "loading",
        "pipeline_wired": PIPELINE_READY,
    }



def _document_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(403, str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(404, str(exc.args[0]))
    return HTTPException(400, str(exc))


@app.post("/api/documents")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    try:
        content = await file.read()
        return await asyncio.to_thread(
            get_assistant_service().ingest_document,
            file.filename or "upload.pdf", content, session_id,
        )
    except (ValueError, KeyError, PermissionError) as exc:
        raise _document_error(exc) from exc


@app.get("/api/documents")
async def list_documents(session_id: str) -> list[dict]:
    return get_assistant_service().list_documents(session_id)


@app.get("/api/documents/{document_id}")
async def get_document(document_id: str, session_id: str) -> dict:
    try:
        return get_assistant_service().get_document(document_id, session_id)
    except (ValueError, KeyError, PermissionError) as exc:
        raise _document_error(exc) from exc


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, session_id: str) -> dict[str, str]:
    try:
        get_assistant_service().delete_document(document_id, session_id)
        return {"status": "deleted"}
    except (ValueError, KeyError, PermissionError) as exc:
        raise _document_error(exc) from exc

@app.post("/api/quiz")
async def quiz(req: QuizRequest) -> dict:
    """Generate a validated MCQ quiz from one of three sources. A controlled
    failure (HTTP 422) is returned instead of a malformed quiz when
    generation/validation can't produce a valid result after one retry."""
    if not _model_loaded:
        raise HTTPException(503, "Models still loading, try again shortly.")

    async with _generate_lock:
        result = await asyncio.to_thread(
            get_assistant_service().quiz,
            req.topic, req.count, req.source_mode, req.document_ids, req.session_id,
        )
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result


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
        topic_used, routing_path = await _resolve_topic(req)
        answer, query_id = await asyncio.to_thread(
            _run_chat_sync, topic_used, req.question, req.session_id, routing_path,
            req.document_ids,
        )

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
            topic_used, routing_path = await _resolve_topic(req)

            import queue
            import threading

            q: "queue.Queue[object]" = queue.Queue()
            DONE = object()

            def producer():
                try:
                    for item in _run_chat_stream_sync(
                        topic_used, req.question, req.session_id, routing_path,
                        req.document_ids,
                    ):
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

        yield _sse_event(
            json.dumps({'query_id': query_id, 'topic_used': topic_used}), event="done"
        )

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest) -> dict[str, str]:
    """Record sanitized evaluation events for the flagged response(s).

    Task 38 (external review, workspace.md Entry 055): this used to persist
    a raw asdict(QueryTrace) per resolved query_id -- including the full
    prompt and raw model output, unredacted for every path except the
    document/quiz ones that already redact -- into logs/feedback.jsonl on
    local disk. Every event now goes through EvaluationService's sanitizer
    instead: metadata only by default, with query/answer text retained only
    because this specific flow is the user's own explicit, consented
    feedback submission (Task 21/23's existing consent UI), never for
    passive telemetry. The raw client-side conversation payload is
    deliberately no longer persisted at all -- it was never on the
    allowlist and isn't needed to review one flagged exchange.
    """
    trace_ids = [req.query_id]
    if req.scope == "conversation" and req.conversation:
        trace_ids.extend(
            message.get("query_id") for message in req.conversation
            if isinstance(message, dict) and message.get("query_id")
        )

    seen = set()
    recorded_any = False
    for query_id in trace_ids:
        if not query_id or query_id in seen:
            continue
        seen.add(query_id)
        trace = get_trace(query_id)
        if trace is not None:
            _evaluation_service.record_trace(
                trace, activity="ask", session_id=req.session_id,
                feedback=req.reaction, feedback_comment=req.comment,
                include_text=True,
            )
            recorded_any = True

    if not recorded_any:
        # The trace was evicted (bounded SESSION_TRACES cache, or a server
        # restart) -- still record the feedback signal itself rather than
        # silently dropping it, just without the metadata a trace would add.
        # Task 45 (external review 2026-10-08): this used to loop over
        # _evaluation_service.sinks and call sink.record() directly, which
        # skipped record_trace()/record_event()'s per-sink try/except --
        # a broken sink here would have turned this feedback submission
        # into an unhandled exception instead of a logged, isolated
        # failure. It also stamped a naive (non-UTC) timestamp while every
        # other EvaluationEvent uses an explicit UTC offset. Both fixed by
        # routing through the same record_event() the normal path uses.
        _evaluation_service.record_event(EvaluationEvent(
            event_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            query_id=req.query_id,
            session_hash=session_hash(req.session_id),
            activity="ask",
            source_mode="",
            feedback=req.reaction,
            feedback_comment=req.comment,
        ))

    return {"status": "ok"}


if __name__ == "__main__":
    import socket
    import uvicorn

    def assert_port_available(host: str, port: int) -> None:
        """Fail before lifespan startup can allocate another GPU model copy."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError as exc:
                raise SystemExit(
                    f"AUM API endpoint {host}:{port} is unavailable. "
                    "Check /api/health and reuse the existing runtime."
                ) from exc

    host = os.environ.get("AUM_API_HOST", "127.0.0.1")
    port = int(os.environ.get("AUM_API_PORT", "8000"))
    assert_port_available(host, port)
    log.info(f"Starting AUM API server on fixed endpoint {host}:{port}")
    uvicorn.run(app, host=host, port=port)
