# main_app.py

import gradio as gr
from typing import List, Dict, Tuple
from pathlib import Path

from modes.faculty_mode import init_faculty_mode, faculty_chat_handler,faculty_pipeline
from modes.research_mode import init_research_mode, research_chat_handler
from modes.exam_mode import init_qa_mode, exam_chat_handler, exam_qa_pipeline  # if you have it


BASE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = BASE_DIR / "datasets"

COS_DATA_FILE = DATASETS_DIR / "COS_research_data" / "cos_data.jsonl"
COS_EMB_DIR   = DATASETS_DIR / "COS_research_data" / "emb_store"

FACULTY_DATA_FILE = DATASETS_DIR / "faculty_dataset" / "faculty_data.json"
FACULTY_EMB_DIR   = DATASETS_DIR / "faculty_dataset" / "emb_store"


# =========================================================
# Init all modes once at startup
# =========================================================
def init_all_modes():
    # 1) Faculty mode
    init_faculty_mode(
        data_file=str(FACULTY_DATA_FILE),
        out_dir=str(FACULTY_EMB_DIR),
        rebuild_store=False,   # True if you change JSON and want fresh embeddings
    )

    # 2) COS Research mode
    init_research_mode(
        data_file=str(COS_DATA_FILE),
        out_dir=str(COS_EMB_DIR),
        rebuild_store=False,
    )

    # 3) Exam QA mode (no datasets, only Phi model)
    init_qa_mode(use_phi15=False)  # or True for smaller model


# =========================================================
# Router for Gradio
# =========================================================
def gradio_chatbot(query, mode, history, memory):
    print(f"\n🚦 Mode Selected: {mode}")

    if mode == "Faculty Research Mode":
        updated_history, updated_memory = faculty_pipeline(query, history, memory)
        return updated_history, updated_memory

    elif mode == "Research Mode":
        updated_history, updated_memory = research_chat_handler(query, history, memory)
        return updated_history, updated_memory

    elif mode == "Exam QA Mode":
        updated_history = exam_qa_pipeline(query, history)
        # QA mode doesn’t use memory; just pass it through unchanged
        return updated_history, memory

    else:
        # Fallback, should not normally happen
        answer = "Unknown mode selected."
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        return history, memory


# =========================================================
# Gradio UI (same look as your v1)
# =========================================================
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧠 AI Teaching Assistant\nChoose a mode and ask your question")

        mode_dropdown = gr.Dropdown(
            choices=[
                "Faculty Research Mode",
                "Research Mode",
                "Exam QA Mode",
            ],
            value="Faculty Research Mode",
            label="Mode",
        )

        chatbox = gr.Chatbot(type="messages", label="Chat History")
        query_input = gr.Textbox(label="Ask your question...")
        clear = gr.Button("Clear Chat")

        history_state = gr.State([])   # full chat history
        memory_state = gr.State([])    # only for faculty / research if you want

        # On submit → route
        query_input.submit(
            gradio_chatbot,
            inputs=[query_input, mode_dropdown, history_state, memory_state],
            outputs=[chatbox, memory_state],
        )

        # Clear textbox after submit
        query_input.submit(lambda _: "", inputs=query_input, outputs=query_input)

        # Clear button resets chat + memory
        clear.click(lambda: ([], []), outputs=[chatbox, memory_state])

    return demo


if __name__ == "__main__":
    init_all_modes()
    app = build_ui()
    app.launch(debug=True, share=True)
