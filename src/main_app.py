# main_app.py

import gradio as gr
from typing import List, Dict, Tuple

from modes.faculty_mode import init_faculty_mode, faculty_chat_handler
from modes.research_mode import init_research_mode, research_chat_handler
from modes.exam_mode import exam_chat_handler  # if you have it

# 1) Initialize shared models (Phi-2, CrossEncoder, MiniLM) once
#    and pass them into init_* functions to avoid loading twice.

def gradio_chatbot(
    query: str,
    mode: str,
    history: List[Dict[str, str]],
    memory: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:

    if mode == "Faculty Research Mode":
        return faculty_chat_handler(query, history, memory)

    elif mode == "COS Research Mode":
        return research_chat_handler(query, history, memory)

    elif mode == "Exam QA Mode":
        return exam_chat_handler(query, history, memory)

    else:
        # Fallback
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": "Unknown mode."})
        return history, memory


with gr.Blocks() as demo:
    gr.Markdown("## 🧠 CS-BOT @ AUM\nAsk about faculty, COS research, or exams")

    mode_dropdown = gr.Dropdown(
        choices=["Faculty Research Mode", "COS Research Mode", "Exam QA Mode"],
        value="Faculty Research Mode",
        label="Mode"
    )

    chatbox = gr.Chatbot(type="messages", label="Chat History")
    query_input = gr.Textbox(label="Ask your question...")
    clear = gr.Button("Clear Chat")

    # Single shared history + memory
    history_state = gr.State([])
    memory_state = gr.State([])

    query_input.submit(
        gradio_chatbot,
        inputs=[query_input, mode_dropdown, history_state, memory_state],
        outputs=[chatbox, memory_state],
    )

    query_input.submit(lambda _: "", inputs=query_input, outputs=query_input)

    clear.click(lambda: ([], []), outputs=[chatbox, history_state])

demo.launch(debug=True, share=True)
