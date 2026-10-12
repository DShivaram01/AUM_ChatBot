"""Generation baseline inside the experiment harness: production
pipeline.quiz.generate_quiz, unchanged, with production evidence selection
(top-6 chunks for the intent topic), on MistralLM. This is the reference arm
every generation experiment is compared with (and is cross-checked against the
live-server Phase 1 run).

    python -m experiments.quiz_bench.run_gen_baseline --out results/dsa_gen_baseline.json
"""

import argparse
import json
import time

from experiments.quiz_bench import gen_harness as H
from experiments.quiz_bench.fixture import gold


def production_quiz(lm, topic, count):
    import pipeline.quiz as pq
    hits = H.retrieve(topic, 6)
    ctx = [c["text"] for c, _ in hits]
    evmap = {i: c["chunk_id"] for i, (c, _) in enumerate(hits, 1)}
    orig = pq.generate_streaming
    pq.generate_streaming = lm.production_streaming_shim()
    try:
        quiz, errors = pq.generate_quiz(topic, count, lm.tok, lm.model, "QZB", context_chunks=ctx,
                                        evidence_map=evmap, require_evidence=True)
    finally:
        pq.generate_streaming = orig
    return quiz, errors, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--fake", action="store_true", help="dry run with a fake model (no GPU)")
    a = ap.parse_args()
    from pipeline.quiz_intent import detect_quiz_intent
    lm = FakeLM() if a.fake else H.MistralLM()
    g = gold()
    res = {"arm": "production_baseline", "load_seconds": getattr(lm, "load_seconds", 0), "requests": {}}
    private = {}
    for key, req in g["quiz_requests"].items():
        if a.only and key not in a.only.split(","):
            continue
        intent = detect_quiz_intent(req["prompt"], has_attached_document=True)
        before = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens, lm.log.truncated_prompts)
        t = time.perf_counter()
        quiz, errors, hits = production_quiz(lm, intent.topic, intent.count)
        secs = time.perf_counter() - t
        qs = quiz["questions"] if quiz else []
        rec = {"requested": intent.count, "valid": quiz is not None, "errors": errors, "seconds": round(secs, 1),
               "calls": lm.log.calls - before[0], "input_tokens": lm.log.input_tokens - before[1],
               "output_tokens": lm.log.output_tokens - before[2], "truncated_prompts": lm.log.truncated_prompts - before[3],
               "evidence_chunks": [c["chunk_id"] for c, _ in hits], "top_score": round(hits[0][1], 3),
               **H.score_quiz(qs, req, ["red-black", "red black"] if key == "F" else None)}
        rec["seconds_per_question"] = round(secs / max(1, len(qs)), 2)
        res["requests"][key] = rec
        private[key] = qs
        print(key, json.dumps({k: rec.get(k) for k in ("valid", "n_questions", "seconds", "calls", "truncated_prompts",
                                                        "near_duplicate_pairs", "n_gold_concepts", "cited_chapters",
                                                        "lexically_supported", "copied_embedded_questions",
                                                        "questions_mentioning_topic")}), flush=True)
    res["gpu_peak_gib"] = H.gpu_peak_gib()
    json.dump(res, open(a.out, "w"), indent=1)
    if not a.fake:
        print("private outputs ->", H.save_private("production_baseline", private))


class FakeLM:
    """Deterministic stand-in so the harness can be tested without a GPU."""
    def __init__(self):
        from transformers import AutoTokenizer
        import config
        self.tok = AutoTokenizer.from_pretrained(config.LLM_ID)
        self.model = None
        self.log = H.CallLog()
        self.n = 0

    def ntok(self, t):
        return len(self.tok(t, add_special_tokens=False)["input_ids"])

    def generate(self, prompt, max_new_tokens, avoid_texts=None, truncate=False, **_):
        from pipeline.answer import _PROMPT_MAX_CHARS
        if truncate and len(prompt) > _PROMPT_MAX_CHARS:
            self.log.truncated_prompts += 1
        self.n += 1
        self.log.calls += 1
        self.log.input_tokens += self.ntok(prompt)
        return json.dumps({"question": f"Fake question {self.n}?", "options": ["a", "b", "c", "d"],
                           "correct_index": 0, "explanation": "x", "evidence_refs": [1]})

    def production_streaming_shim(self):
        def shim(prompt, tokenizer, model, query_id="Q", max_new_tokens=300, trace=None, avoid_texts=None):
            yield self.generate(prompt, max_new_tokens, avoid_texts=avoid_texts, truncate=True)
        return shim


if __name__ == "__main__":
    main()
