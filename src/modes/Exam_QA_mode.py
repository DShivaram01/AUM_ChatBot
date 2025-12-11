"""
qa_mode.py

Multiple-choice Exam QA mode for your CS-BOT.

Provides:
- init_qa_mode(...)     -> load Phi model for MCQ generation
- exam_qa_pipeline(...) -> generate 10 MCQs with answers from a given topic

Intended usage (from main_app.py or a notebook):

    from qa_mode import init_qa_mode, exam_qa_pipeline

    init_qa_mode(use_phi15=False)  # or True if you want smaller/faster phi-1_5

    history = []
    history = exam_qa_pipeline("Data Structures: Stacks and Queues", history)
"""

from __future__ import annotations

from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================================================================
# GLOBALS
# =====================================================================

phi_model = None
phi_tokenizer = None
PHI_MODEL_ID_DEFAULT = "microsoft/phi-2"
PHI_MODEL_ID_SMALL   = "microsoft/phi-1_5"


# =====================================================================
# INITIALIZATION
# =====================================================================

def init_qa_mode(
    use_phi15: bool = False,
    phi_model_id: Optional[str] = None,
) -> None:
    """
    Initialize the Exam QA mode by loading a Phi model.

    Args:
        use_phi15:
            - False (default): use microsoft/phi-2 (better quality)
            - True: use microsoft/phi-1_5 (smaller & a bit faster)
        phi_model_id:
            - Optional manual override for the model ID.
    """
    global phi_model, phi_tokenizer

    if phi_model_id is not None:
        chosen_id = phi_model_id
    else:
        chosen_id = PHI_MODEL_ID_SMALL if use_phi15 else PHI_MODEL_ID_DEFAULT

    print(f"🧠 Loading Phi model for QA mode: {chosen_id}")
    phi_tokenizer = AutoTokenizer.from_pretrained(chosen_id)
    phi_model = AutoModelForCausalLM.from_pretrained(
        chosen_id,
        device_map="auto",
        torch_dtype="auto",
    )

    # Warmup to avoid first-call latency
    _ = phi_model.generate(
        **phi_tokenizer("warmup", return_tensors="pt").to(phi_model.device),
        max_new_tokens=1,
    )
    print("✅ QA mode initialized (Phi warmup done).")


# =====================================================================
# EXAM QA PIPELINE
# =====================================================================

def exam_qa_pipeline(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Exam QA pipeline:
    Given a topic or content description in `query`,
    generate exactly 10 multiple-choice questions (MCQs) with answers.

    Args:
        query:   Topic / short notes / chapter name, etc.
        history: Existing chat history (list of {'role', 'content'} dicts)

    Returns:
        updated_history: history + this user query + MCQ response
    """
    global phi_model, phi_tokenizer

    if history is None:
        history = []

    if phi_model is None or phi_tokenizer is None:
        raise RuntimeError(
            "QA mode not initialized. Call init_qa_mode(...) before using exam_qa_pipeline()."
        )

    print(f"\n🧠 Incoming Exam QA Query: {query}")

    prompt = f"""You are a university teaching assistant helping students prepare for exams.

Topic or Input:
{query}

Task:
Generate exactly 10 multiple choice questions (MCQs) with answers.
The questions should:
- Cover the main ideas and important details in the topic.
- Have 4 options (A, B, C, D) for each question.
- Include the correct option letter on a separate line.

Format strictly as:

1. Question text...
A. Option text...
B. Option text...
C. Option text...
D. Option text...
Answer: <Correct option letter>

2. Question text...
A. ...
B. ...
C. ...
D. ...
Answer: <Correct option letter>

Continue until 10 questions are generated.
Do not add any extra commentary.

Begin below:
"""

    inputs = phi_tokenizer(prompt, return_tensors="pt").to(phi_model.device)
    outputs = phi_model.generate(**inputs, max_new_tokens=700)
    response = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Keep only the content after "Begin below:"
    if "Begin below:" in response:
        response = response.split("Begin below:", 1)[1].strip()

    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]
    return updated_history
