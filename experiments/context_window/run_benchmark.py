"""How far past the app's current hard caps -- _PROMPT_MAX_CHARS=3200
(pipeline/answer.py) and the tokenizer's max_length=4096 -- can the real
Mistral-7B-Instruct-v0.3 actually go on this machine's GPU?

Loads the real production model (models.loader.load_llm(), same dtype/
attention config the server uses) and runs the exact summarize prompt
template (pipeline/summarize.generate_summary_basic) against the real
8k/25k-token fixtures (experiments/context_window/fixtures/, built by
build_fixtures.py from real AUM content), WITHOUT the 3200-char
pre-truncation or the 4096-token tokenizer cap -- those are bypassed
here on purpose, to measure the model's actual behavior rather than the
app's current self-imposed ceiling.

Reuses GeneratedOnlyRepetitionGuard so decoding is faithful to
production (same anti-repetition behavior), not a bare generate() call.
Catches CUDA OOM per fixture so one failure doesn't take down the run.
"""

import gc
import json
import time
import traceback
from pathlib import Path

import torch
from transformers import LogitsProcessorList

from models.loader import load_llm
from pipeline.answer import GeneratedOnlyRepetitionGuard

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

_SUMMARIZE_PROMPT = (
    "<s>[INST] Summarize the following document in a few clear "
    "paragraphs. Use only information in the text below; do not add "
    "facts that aren't there.\n\n"
    "<context>\n{context}\n</context> [/INST]"
)


def run_one(label: str, text: str, llm_tok, llm_model) -> dict:
    prompt = _SUMMARIZE_PROMPT.format(context=text)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    # No truncation, no max_length cap -- the point of this run is to see
    # what the real token count is and whether the model/GPU can take it.
    inputs = llm_tok(prompt, return_tensors="pt").to(llm_model.device)
    input_tokens = int(inputs["input_ids"].shape[1])
    tokenize_s = time.time() - t0

    result = {
        "label": label, "input_tokens": input_tokens,
        "tokenize_seconds": round(tokenize_s, 2),
    }

    gen_kw = dict(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        logits_processor=LogitsProcessorList([GeneratedOnlyRepetitionGuard(input_tokens)]),
        eos_token_id=llm_tok.eos_token_id,
        pad_token_id=llm_tok.eos_token_id,
    )

    t0 = time.time()
    try:
        with torch.no_grad():
            out = llm_model.generate(**gen_kw)
        gen_s = time.time() - t0
        output_text = llm_tok.decode(out[0][input_tokens:], skip_special_tokens=True).strip()
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        result.update({
            "status": "ok",
            "generate_seconds": round(gen_s, 2),
            "output_tokens": int(out.shape[1] - input_tokens),
            "peak_gpu_gib": round(peak_gib, 2),
            "output_preview": output_text[:800],
        })
    except torch.cuda.OutOfMemoryError as exc:
        result.update({
            "status": "oom", "generate_seconds": round(time.time() - t0, 2),
            "error": str(exc),
        })
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        result.update({
            "status": "oom", "generate_seconds": round(time.time() - t0, 2),
            "error": str(exc),
        })
    finally:
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    return result


def main():
    print("Loading real Mistral-7B-Instruct-v0.3 (production loader, float16/GPU)...")
    llm_tok, llm_model = load_llm()

    # Smallest first on purpose, not alphabetical ("25k" < "8k" as a
    # string) -- confirm the cheaper run succeeds before risking OOM on
    # the larger one.
    paths = sorted(
        FIXTURES_DIR.glob("doc_*.txt"),
        key=lambda p: int(p.stem.removeprefix("doc_").rstrip("k")),
    )
    results = []
    for path in paths:
        label = path.stem.removeprefix("doc_")
        text = path.read_text(encoding="utf-8")
        print(f"\n=== {label}: {path.name} ({len(text)} chars) ===")
        try:
            result = run_one(label, text, llm_tok, llm_model)
        except Exception as exc:  # pragma: no cover - diagnostic script
            result = {"label": label, "status": "error", "error": repr(exc)}
            traceback.print_exc()
        print(json.dumps({k: v for k, v in result.items() if k != "output_preview"}, indent=2))
        results.append(result)

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results (incl. output previews) written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
