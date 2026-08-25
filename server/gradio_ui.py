"""
server/gradio_ui.py
=====================
The standalone Gradio UI, extracted from backend.py.

Design note (see workspace.md TASK 11 log entry): in backend.py, cos_chat()
and housing_chat() are plain module-level functions that close over ~14
top-level globals (embedder, cos_index, cos_META, llm_tok, llm_model,
H_index, housing_ok, DEBUG_MODE, ...) created by imperative code earlier in
the same flat script. That pattern doesn't survive a module split as-is --
there's no shared enclosing script anymore. Rather than rewrite cos_chat/
housing_chat into fully dependency-injected functions (a bigger behavior-
risk change the task didn't ask for), this module keeps them as
module-level functions reading module-level globals, same as the original,
and adds one explicit init_gradio_ui() setter that main.py calls once
after loading models/data -- the smallest change that lets the split work.
"""

import os
import time

import gradio as gr

import config
from pipeline.memory import logger
from pipeline.classifier import classify_query, next_query_id, SESSION_TRACES, get_trace
from pipeline.retrieval import retrieve_cos_rrf, retrieve_housing_logged
from pipeline.answer import build_cos_answer_streaming, build_housing_answer_streaming, render_trace_md

DEBUG_MODE = os.environ.get("AUM_DEBUG_MODE", "true").lower() == "true"

# ── Runtime state, populated by init_gradio_ui() (see module docstring) ──
embedder = reranker = llm_tok = llm_model = None
cos_index = cos_EMB = cos_META = cos_TEXTS = cos_bm25 = None
H_index = H_EMB = housing_chunks = None
housing_ok = False
_log_path = ""


def init_gradio_ui(
    *, embedder_, reranker_, llm_tok_, llm_model_,
    cos_index_, cos_EMB_, cos_META_, cos_TEXTS_, cos_bm25_,
    H_index_, H_EMB_, housing_chunks_, housing_ok_,
    log_path_,
):
    """Must be called once, after models + data are loaded, before
    _build_gradio_demo() / launch()."""
    global embedder, reranker, llm_tok, llm_model
    global cos_index, cos_EMB, cos_META, cos_TEXTS, cos_bm25
    global H_index, H_EMB, housing_chunks, housing_ok, _log_path

    embedder, reranker, llm_tok, llm_model = embedder_, reranker_, llm_tok_, llm_model_
    cos_index, cos_EMB, cos_META, cos_TEXTS, cos_bm25 = (
        cos_index_, cos_EMB_, cos_META_, cos_TEXTS_, cos_bm25_
    )
    H_index, H_EMB, housing_chunks, housing_ok = H_index_, H_EMB_, housing_chunks_, housing_ok_
    _log_path = log_path_


import re as _re
_SELECTION_RE = _re.compile(r"^\s*(\d{1,2})\s*$")


def cos_chat(message, history, pending_cands):
    message = message.strip()
    qid = next_query_id("COS")
    logger.info(f"\n{'='*65}")
    logger.info(f"[{qid}] NEW COS QUERY: {repr(message)}")

    if not message:
        yield "Please type a question about AUM research projects.", pending_cands, "", qid
        return

    sel_m = _SELECTION_RE.match(message)
    if sel_m and pending_cands:
        sel_n = int(sel_m.group(1))
        if 1 <= sel_n <= len(pending_cands):
            selected = pending_cands[sel_n - 1]
            selected["_selection_n"] = sel_n
            m = selected["meta"]
            logger.info(f"[{qid}] User selected #{sel_n}: {m.get('title','')[:50]}")
            fake_qinfo = {
                "type": "TYPE_TOPIC", "person_hints": [], "person_hint": None,
                "year_hint": None, "dept_hint": None, "is_broad": False,
            }
            for partial, _, trace in build_cos_answer_streaming(
                f"Tell me about: {m.get('title','')}", [], fake_qinfo,
                llm_tok, llm_model, qid, selected_cand=selected,
            ):
                debug_md = render_trace_md(trace) if DEBUG_MODE else ""
                yield partial, [], debug_md, qid
            return
        yield f"Please enter a number between 1 and {len(pending_cands)}.", pending_cands, "", qid
        return

    qinfo = classify_query(message)
    cands = retrieve_cos_rrf(
        message, qinfo, embedder, cos_index, cos_EMB,
        cos_META, cos_TEXTS, cos_bm25, reranker, query_id=qid,
    )
    new_pending = []
    for partial, returned_cands, trace in build_cos_answer_streaming(
        message, cands, qinfo, llm_tok, llm_model, qid
    ):
        if returned_cands is not None:
            new_pending = returned_cands
        debug_md = render_trace_md(trace) if DEBUG_MODE else ""
        yield partial, new_pending, debug_md, qid


def housing_chat(message, history, _pending):
    message = message.strip()
    qid = next_query_id("HSG")
    logger.info(f"\n{'='*65}")
    logger.info(f"[{qid}] NEW HOUSING QUERY: {repr(message)}")

    if not message:
        yield "Please type a question about AUM Housing policy.", [], "", qid
        return
    if not housing_ok:
        yield f"Housing PDF not found: {config.HOUSING_PDF}", [], "", qid
        return

    started = time.time()
    hits = retrieve_housing_logged(
        message, embedder, H_index, H_EMB, housing_chunks, query_id=qid
    )
    search_ms = (time.time() - started) * 1000
    for partial, trace in build_housing_answer_streaming(
        message, hits, llm_tok, llm_model, qid, search_ms=search_ms
    ):
        debug_md = render_trace_md(trace) if DEBUG_MODE else ""
        yield partial, [], debug_md, qid


# ── Gradio UI (backend.py:2059-2204) ───────────────────────────────────

COS_EXAMPLES = [
    "List 3 projects related to Dr. Sutanu Bhattacharya",
    "What Biology projects were presented in 2024?",
    "Tell me about projects mentored by Jerome Goddard",
    "Name a research project related to protein sequences",
    "What research has been done in the Mathematics department?",
]
HOUSING_EXAMPLES = [
    "What are the quiet hours in the residence halls?",
    "Can I have a pet in my room?",
    "What happens if there is a fire alarm?",
    "What is the guest policy for overnight visitors?",
    "Are candles or incense allowed in dorm rooms?",
]

CSS = (
    ".gradio-container{font-family:Georgia,serif;}"
    ".tab-nav button{font-size:18px;font-weight:600;}"
    "footer{display:none!important;}"
    "#debug-panel{background:#1a1a2e;color:#eee;font-size:0.82rem;"
    "padding:12px;border-radius:6px;overflow-y:auto;max-height:620px;}"
    ".message-wrap .message { font-size: 20px !important; line-height: 1.5; }"
    ".chatbot .p { font-size: 20px !important; }"
)


def _chat_tab(chat_fn, examples, bot_label, placeholder):
    """Builds one full chat tab with optional debug panel."""
    pending_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                height=900,
                label=bot_label,
                placeholder=placeholder,
            )
            with gr.Row():
                msg_box  = gr.Textbox(
                    placeholder="Type your question and press Enter...",
                    show_label=False, scale=5,
                )
                send_btn = gr.Button("Ask", scale=1, variant="primary")
            gr.Examples(examples=examples, inputs=msg_box)

        with gr.Column(scale=1, visible=DEBUG_MODE) as debug_col:
            gr.Markdown("### Pipeline Trace")
            debug_panel = gr.Markdown(
                "_Ask a question to see the full pipeline trace here._",
                elem_id="debug-panel",
            )

    def _submit(message, history, pending):
        history = history or []
        base_history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": ""},
        ]
        new_pending = pending
        debug_out   = ""
        for partial, np, dbg, _query_id in chat_fn(message, history, pending):
            new_pending = np  if np  is not None else new_pending
            debug_out   = dbg if dbg is not None else debug_out
            base_history[-1]["content"] = partial
            yield base_history, base_history, new_pending, "", debug_out

    for trigger in [send_btn.click, msg_box.submit]:
        trigger(
            _submit,
            inputs  = [msg_box, chatbot, pending_state],
            outputs = [chatbot, chatbot, pending_state, msg_box, debug_panel],
        )


def _make_trace_inspector():
    """Standalone tab that lets you browse all SESSION_TRACES from the
    current session without touching the log file."""
    gr.Markdown("### Session Trace Inspector")
    gr.Markdown(
        "Select a query ID to see its full pipeline trace. "
        "Updates automatically after each query."
    )

    def _get_trace_ids():
        if not SESSION_TRACES:
            return ["(no queries yet)"]
        return [f"{t.query_id} — {t.query[:45]}" for t in reversed(SESSION_TRACES)]

    def _show_trace(selection):
        if not selection or selection.startswith("("):
            return "_No trace selected._"
        qid = selection.split(" — ")[0].strip()
        t   = get_trace(qid)
        return render_trace_md(t) if t else "_Trace not found._"

    trace_dd = gr.Dropdown(
        choices    = _get_trace_ids(),
        label      = "Query",
        interactive= True,
    )
    refresh_btn   = gr.Button("Refresh list")
    trace_display = gr.Markdown("_Select a query above._")

    refresh_btn.click(
        fn      = lambda: gr.update(choices=_get_trace_ids()),
        outputs = [trace_dd],
    )
    trace_dd.change(
        fn      = _show_trace,
        inputs  = [trace_dd],
        outputs = [trace_display],
    )


def _build_gradio_demo():
    with gr.Blocks(title="AUM-Chatbot v4.1") as demo:
        gr.Markdown(
            f"# AUM-Chatbot v4.1\n"
            f"COS Research  |  Housing Policy  "
            f"{'— DEBUG MODE ON' if DEBUG_MODE else ''}\n\n"
            f"Log: `{_log_path}`"
        )

        with gr.Tabs():
            with gr.Tab("COS Research Symposium"):
                gr.Markdown(
                    "Ask about AUM research projects. "
                    "For broad questions I will show a list — reply with a number for the full summary."
                )
                _chat_tab(cos_chat, COS_EXAMPLES,
                          "COS Research Assistant", "Ask about AUM research projects...")

            with gr.Tab("Housing and Community Standards"):
                gr.Markdown("Ask about AUM Housing policies.")
                _chat_tab(housing_chat, HOUSING_EXAMPLES,
                          "Housing Policy Assistant", "Ask about AUM housing policies...")

            with gr.Tab("Pipeline Inspector"):
                _make_trace_inspector()
    return demo


def launch():
    """Builds and blocks on the standalone Gradio UI (backend.py's
    `if __name__ == "__main__":` block, backend.py:2219-2227). Requires
    init_gradio_ui() to have been called first."""
    logger.info("[Gradio] Launching UI...")
    demo = _build_gradio_demo()
    demo.launch(
        server_name=os.environ.get("AUM_GRADIO_HOST", "127.0.0.1"),
        server_port=7860,
        share=False,
        inbrowser=False,
        css=CSS,
    )
