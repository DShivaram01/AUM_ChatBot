"""Experiment 7: structured JSON generation, compared at the decoding layer.

    python -m experiments.quiz_bench.structured --arms prompt lmfe outlines --out results/dsa_structured.json

Every arm fills the SAME blueprint slots in batches of 5 (batch_gen.SlotFiller:
same prompt, same validation, same targeted repair); only the decoding
mechanism differs:
  prompt    free generation + production JSON repair/parse (control)
  lmfe      lm-format-enforcer JsonSchemaParser -> prefix_allowed_tokens_fn on
            model.generate (schema-constrained decoding, same model object)
  outlines  Outlines 1.3 Generator over the same loaded transformers model with
            a Pydantic output_type (schema-constrained decoding)
Schema: one canonical Pydantic model (QuizBatch) built per batch so the
question count is exact (min_items == max_items == batch size).

LMQL (template/query programming; loads its own model copy) runs in a
separate process: structured_lmql.py. Instructor needs an OpenAI-compatible
or provider client and has no in-process transformers backend; its
validate-and-retry role is what SlotFiller already does with Pydantic-equivalent
checks -- recorded as not applicable to this backend, not benchmarked.
"""

import argparse
import json
import time
from typing import List

from pydantic import BaseModel, Field, conlist

from experiments.quiz_bench import gen_harness as H
from experiments.quiz_bench.batch_gen import SlotFiller, make_batches, OUT_TOKENS_PER_Q
from experiments.quiz_bench.blueprint import build
from experiments.quiz_bench.fixture import gold


class QuizQuestion(BaseModel):
    """Canonical quiz item (all arms are validated against the same rules)."""
    slot: int
    question: str = Field(min_length=10, max_length=400)
    options: conlist(str, min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str = Field(max_length=500)
    evidence_refs: conlist(int, min_length=1, max_length=3)


def batch_model(n: int):
    class QuizBatch(BaseModel):
        questions: conlist(QuizQuestion, min_length=n, max_length=n)
    return QuizBatch


def lmfe_decoder(lm):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    def decode(group):
        parser = JsonSchemaParser(batch_model(len(group)).model_json_schema())
        return {"prefix_allowed_tokens_fn": build_transformers_prefix_allowed_tokens_fn(lm.tok, parser)}
    return decode


class OutlinesLM:
    """Routes SlotFiller's generate() through an Outlines Generator over the
    already-loaded model; keeps the same call/token/time accounting."""

    def __init__(self, lm):
        import outlines
        self.lm, self.log, self.ntok, self.tok = lm, lm.log, lm.ntok, lm.tok
        self.model = outlines.from_transformers(lm.model, lm.tok)
        self._gens = {}

    def generate(self, prompt, max_new_tokens, avoid_texts=None, n_slots=1, **_):
        import outlines
        n = prompt.count("- slot ")
        if n not in self._gens:
            self._gens[n] = outlines.Generator(self.model, batch_model(max(1, n)))
        t = time.perf_counter()
        out = self._gens[n](prompt, max_new_tokens=max_new_tokens, do_sample=False)
        secs = time.perf_counter() - t
        self.log.calls += 1
        self.log.input_tokens += self.ntok(prompt)
        self.log.output_tokens += self.ntok(out)
        self.log.seconds += secs
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["prompt", "lmfe", "outlines"])
    ap.add_argument("--requests", default="B,C,D,E")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pipeline.quiz_intent import detect_quiz_intent
    lm = H.MistralLM()
    g = gold()
    chunks, _ = H.document()
    text = {c["chunk_id"]: c["text"] for c in chunks}
    out, private = {"arms": {}, "instructor": "not applicable: no in-process transformers backend (needs an OpenAI-compatible/provider client); its validation+retry role duplicates SlotFiller", "lmql": "see structured_lmql.py (own model copy, separate process)"}, {}
    for arm in a.arms:
        gen_lm = OutlinesLM(lm) if arm == "outlines" else lm
        decode = lmfe_decoder(lm) if arm == "lmfe" else None
        out["arms"][arm], private[arm] = {}, {}
        for key in a.requests.split(","):
            req = g["quiz_requests"][key]
            intent = detect_quiz_intent(req["prompt"], has_attached_document=True)
            bp = build(req["scope"] if req["scope"] == "whole" else "topic", intent.topic, intent.count, lm.ntok)
            b = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens, lm.log.seconds)
            t0 = time.perf_counter()
            filler = SlotFiller(gen_lm, decode)
            try:
                qs = filler.fill(bp.slots, make_batches(bp.slots, "b5", lm.ntok, text), t0)
                err = None
            except Exception as e:                       # framework incompatibility is a result, not a crash
                qs, err = [], f"{type(e).__name__}: {str(e)[:300]}"
            secs = time.perf_counter() - t0
            gen_tok = lm.log.output_tokens - b[2]
            rec = {"planned_slots": len(bp.slots), "complete": len(qs) == len(bp.slots), "error": err,
                   "seconds": round(secs, 1), "calls": lm.log.calls - b[0], "input_tokens": lm.log.input_tokens - b[1],
                   "output_tokens": gen_tok, "decode_tokens_per_second": round(gen_tok / max(1e-9, lm.log.seconds - b[3]), 1),
                   **{k: v for k, v in filler.stats.items() if k != "slot_attempts"}, **H.score_quiz(qs, req)}
            rec["first_pass_valid_rate"] = round(1 - rec["invalid_items"] / max(1, len(bp.slots) + rec["repairs"]), 3)
            out["arms"][arm][key] = rec
            private[arm][key] = qs
            print(f"[{arm}] {key}", json.dumps({k: rec.get(k) for k in ("planned_slots", "n_questions", "complete", "error",
                                                                       "seconds", "calls", "parse_failures", "invalid_items",
                                                                       "repairs", "decode_tokens_per_second", "lexically_supported",
                                                                       "near_duplicate_pairs")}), flush=True)
    out["gpu_peak_gib"] = H.gpu_peak_gib()
    json.dump(out, open(a.out, "w"), indent=1)
    print("private outputs ->", H.save_private("structured", private))


if __name__ == "__main__":
    main()
