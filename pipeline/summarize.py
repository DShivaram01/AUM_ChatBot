"""Basic, single-shot document summarization.

Same "basic" philosophy as pipeline.quiz's generate_quiz_basic(): no
map-reduce, no validation, no corrections. One cheap Mistral call
classifies whether a chat message is a summarize-the-attached-document
request, and (if so) one Mistral call produces the summary over as much
of the document as fits the existing prompt-length safety net
(pipeline.answer._truncate_prompt) -- the same net document_chat() and
quiz already rely on.

Map-reduce was deliberately NOT used here. exp/quiz-map-reduce (commit
55c05eb) benchmarked an LLM map-per-chunk step for quiz concept coverage
against real Mistral: most "map" calls (6-7 out of 9-10) failed to parse
into usable output, and the arm still covered fewer chapters/concepts
than a simpler no-LLM baseline despite costing 109-120s of extra calls.
A per-chunk summarize call is the same shape of request, so the same
reliability risk applies -- single-shot stays simpler and avoids it.
"""

from __future__ import annotations

import torch

from pipeline.answer import generate_streaming

_SUMMARIZE_INTENT_PROMPT = (
    "<s>[INST] Reply with exactly one word, YES or NO, nothing else. "
    "Does this message ask to summarize an attached document (e.g. "
    "\"summarize this\", \"give me a summary of the PDF\", \"what is "
    "this document about\")?\n\n"
    "Message: {question} [/INST]"
)


def classify_summarize_intent(question: str, llm_tok, llm_model) -> bool:
    """Ask Mistral whether this message is a summarize-the-document
    request. Greedy, 5 output tokens -- same cheap-classifier shape as
    pipeline.classifier._classify_topic_llm. Defaults to False (not a
    summarize request) on anything other than a clear "yes": a false
    positive here would silently turn an ordinary document question
    into a full-document summary instead of answering it.
    """
    prompt = _SUMMARIZE_INTENT_PROMPT.format(question=question)
    inputs = llm_tok(
        prompt, return_tensors="pt", truncation=True, max_length=512,
    ).to(llm_model.device)
    with torch.no_grad():
        out = llm_model.generate(
            **inputs, max_new_tokens=5, do_sample=False,
            eos_token_id=llm_tok.eos_token_id, pad_token_id=llm_tok.eos_token_id,
        )
    decoded = llm_tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip().lower()
    return decoded.startswith("yes")


def generate_summary_basic(chunks: list[str], llm_tok, llm_model, query_id: str) -> str:
    """Single-shot document summary: one Mistral call over the joined
    document chunks, no map-reduce, no validation. generate_streaming's
    own prompt-length safety net truncates (preserving the closing
    "[/INST]") if the document doesn't fit -- the summary then simply
    covers only what made it into the prompt, same as quiz's grounded
    context handling.
    """
    context = "\n\n".join(chunks) if chunks else "(no document text available)"
    prompt = (
        "<s>[INST] Summarize the following document in a few clear "
        "paragraphs. Use only information in the text below; do not add "
        "facts that aren't there.\n\n"
        f"<context>\n{context}\n</context> [/INST]"
    )
    full = ""
    for partial in generate_streaming(
        prompt, llm_tok, llm_model,
        query_id=f"{query_id}-summary", max_new_tokens=400,
    ):
        full = partial
    return full.strip()
