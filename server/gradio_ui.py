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
import gradio as gr

from pipeline.memory import logger, _log_path
from pipeline.classifier import SESSION_TRACES, get_trace
from pipeline.answer import render_trace_md
from core.orchestrator import get_assistant_service

DEBUG_MODE = os.environ.get("AUM_DEBUG_MODE", "true").lower() == "true"


def cos_chat(message, history, pending_cands):
    """Debug-client adapter; AssistantService owns chat execution."""
    yield from get_assistant_service().cos_chat(message, history, pending_cands)


def housing_chat(message, history, pending_cands):
    """Debug-client adapter; AssistantService owns chat execution."""
    yield from get_assistant_service().housing_chat(message, history, pending_cands)


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
