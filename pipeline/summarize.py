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


# Exploratory formula from exp/context-window-benchmark (commits
# 5ac8706/be8b0b8): max(400, input_tokens // 15), locked in here after
# it generated a summary that stopped on its own (EOS) well under
# budget at every tested size -- 8k/12k/14k/16k input tokens all used
# 46-80% of their allotted budget, never truncated. The fixed 400 the
# first pass used DID truncate mid-sentence on an 8k-token document, so
# this isn't a theoretical improvement. Not re-derived from first
# principles -- just the one combination that was actually measured.
_OUTPUT_TOKENS_PER_INPUT_TOKEN = 15
_MIN_OUTPUT_TOKENS = 400


_SENTENCE_ENDERS = ".!?"


def _trim_to_last_sentence(text: str) -> str:
    """If text doesn't end on a sentence boundary -- the signature of
    being cut off mid-sentence by the output-token budget running out --
    trim back to the last complete sentence. Leaves text unchanged if no
    sentence-ending punctuation exists anywhere, rather than deleting
    everything; this is a safety net, not a guarantee the content itself
    is complete (a trimmed summary may still be missing what would have
    come after the cut).
    """
    stripped = text.rstrip()
    if not stripped or stripped[-1] in _SENTENCE_ENDERS:
        return stripped
    last_end = max((stripped.rfind(ch) for ch in _SENTENCE_ENDERS), default=-1)
    if last_end == -1:
        return stripped
    return stripped[:last_end + 1]


def generate_summary_basic(chunks: list[str], llm_tok, llm_model, query_id: str) -> str:
    """Single-shot document summary: one Mistral call over the joined
    document chunks, no map-reduce, no JSON validation. generate_streaming's
    own prompt-length safety net truncates (preserving the closing
    "[/INST]") if the document doesn't fit -- the summary then simply
    covers only what made it into the prompt, same as quiz's grounded
    context handling.

    Two things guard against a mid-word cutoff, found live on a
    heterogeneous test document that pushed the model toward listing
    every topic individually instead of writing a few paragraphs: the
    prompt explicitly tells it not to enumerate, and _trim_to_last_sentence()
    below catches it if that instruction doesn't fully work. Neither one
    is validation in the quiz/document-JSON sense -- the raw text is
    still returned as-is otherwise, just with an explicit instruction up
    front and a sentence-boundary safety net on the way out.
    """
    context = "\n\n".join(chunks) if chunks else "(no document text available)"
    prompt = (
        "<s>[INST] Summarize the following document in no more than 5-6 "
        "clear paragraphs. Use only information in the text below; do "
        "not add facts that aren't there. Do NOT list or enumerate every "
        "item, topic, or section individually -- group similar things "
        "together and describe them collectively instead.\n\n"
        f"<context>\n{context}\n</context> [/INST]"
    )
    # Counted once here (cheap -- tokenize_seconds measured 0.01-0.03s in
    # the benchmark) so the budget below can scale with this specific
    # prompt's real size, rather than guessing from character count.
    input_tokens = len(llm_tok.encode(prompt))
    max_new_tokens = max(_MIN_OUTPUT_TOKENS, input_tokens // _OUTPUT_TOKENS_PER_INPUT_TOKEN)

    full = ""
    for partial in generate_streaming(
        prompt, llm_tok, llm_model,
        query_id=f"{query_id}-summary", max_new_tokens=max_new_tokens,
    ):
        full = partial
    return _trim_to_last_sentence(full.strip())
