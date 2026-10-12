"""Phase 10: integrated candidate vs the production baseline, with an ablation.

    python -m experiments.quiz_bench.integrated --out results/dsa_integrated.json

Selected components (from the individual experiments, CS-201 notes):
  chunking   production structured_chunks (best on the notes; the paper's
             recursive-512 gain did not generalise)
  embedder   production MiniLM (Qwen3-0.6B ranked better on both documents but
             needs transformers >= 4.51; left as a separate ablation)
  evidence   blueprint (chapter round-robin / topic above the production
             document floor 0.35; declines unsupported topics), whole chunks,
             token-aware, no 3200-char cap
  structure  Outlines schema-constrained JSON (Pydantic QuizBatch)
  repair     targeted per-slot regeneration, valid items kept

Arms:
  I1_outlines_seq   blueprint + Outlines, 1 slot per call
  I2_outlines_b5    blueprint + Outlines, 5 slots per call
  A1_no_blueprint   I2 without the blueprint: slots dealt over production's
                    top-6 chunks for the intent topic (no floor, no chapter
                    spreading) -- isolates the blueprint's contribution
Baseline (production) and the prompt-only batch arms come from the previous
window (results on exp/quiz-benchmark-baseline and exp/quiz-batch-generation);
decoding is greedy, and the harness reproduced the live server exactly.
"""

import argparse
import json
import time

from experiments.quiz_bench import gen_harness as H
from experiments.quiz_bench.batch_gen import SlotFiller, make_batches
from experiments.quiz_bench.blueprint import OBJECTIVES, Blueprint, Slot, build
from experiments.quiz_bench.fixture import gold
from experiments.quiz_bench.structured import OutlinesLM


def top6_blueprint(topic, n):
    hits = [c for c, _ in H.retrieve(topic, 6)]
    slots = [Slot(i + 1, hits[i % len(hits)]["chunk_id"], H.chapter_of(hits[i % len(hits)]), hits[i % len(hits)]["page"],
                  " > ".join(hits[i % len(hits)]["heading_path"][1:]), OBJECTIVES[(i // len(hits)) % len(OBJECTIVES)],
                  H.chapter_of(hits[i % len(hits)])) for i in range(n)]
    return Blueprint("topic-top6", n, slots, True, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["I1_outlines_seq", "I2_outlines_b5", "A1_no_blueprint"])
    ap.add_argument("--requests", default="A,B,C,D,E,F")
    ap.add_argument("--fake", action="store_true", help="plumbing test: FakeBatchLM instead of Outlines/Mistral")
    a = ap.parse_args()
    from pipeline.quiz_intent import detect_quiz_intent
    if a.fake:
        from experiments.quiz_bench.batch_gen import FakeBatchLM
        lm = gen_lm = FakeBatchLM()
    else:
        lm = H.MistralLM()
        gen_lm = OutlinesLM(lm)
    g = gold()
    chunks, _ = H.document()
    text = {c["chunk_id"]: c["text"] for c in chunks}
    out, private = {"load_seconds": getattr(lm, "load_seconds", 0), "arms": {}}, {}
    for arm in a.arms:
        out["arms"][arm], private[arm] = {}, {}
        batch = "seq" if arm == "I1_outlines_seq" else "b5"
        for key in a.requests.split(","):
            req = g["quiz_requests"][key]
            intent = detect_quiz_intent(req["prompt"], has_attached_document=True)
            bp = top6_blueprint(intent.topic, intent.count) if arm == "A1_no_blueprint" else \
                build(req["scope"] if req["scope"] == "whole" else "topic", intent.topic, intent.count, lm.ntok)
            b = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens)
            t0 = time.perf_counter()
            filler = SlotFiller(gen_lm)
            qs = filler.fill(bp.slots, make_batches(bp.slots, batch, lm.ntok, text), t0) if bp.supported else []
            secs = time.perf_counter() - t0
            rec = {"requested": intent.count, "planned_slots": len(bp.slots), "supported": bp.supported, "note": bp.note,
                   "complete": len(qs) == len(bp.slots), "seconds": round(secs, 1),
                   "calls": lm.log.calls - b[0], "input_tokens": lm.log.input_tokens - b[1], "output_tokens": lm.log.output_tokens - b[2],
                   **{k: v for k, v in filler.stats.items() if k != "slot_attempts"},
                   **H.score_quiz(qs, req, ["red-black", "red black"] if key == "F" else None)}
            out["arms"][arm][key] = rec
            private[arm][key] = {"blueprint": bp.as_dict(), "questions": qs}
            print(f"[{arm}] {key}", json.dumps({k: rec.get(k) for k in ("supported", "planned_slots", "n_questions", "complete", "seconds",
                                                                       "first_result_seconds", "calls", "invalid_items", "repairs",
                                                                       "parse_failures", "near_duplicate_pairs", "n_gold_concepts",
                                                                       "cited_chapters", "lexically_supported", "copied_embedded_questions",
                                                                       "questions_mentioning_topic")}), flush=True)
    out["gpu_peak_gib"] = H.gpu_peak_gib() if not a.fake else None
    json.dump(out, open(a.out, "w"), indent=1)
    if not a.fake:
        print("private outputs ->", H.save_private("integrated", private))


if __name__ == "__main__":
    main()
