"""TASK 42 step 5: benchmark legacy/semantic/hybrid routing against the
labeled gold set (tests/fixtures/routing_gold_set.jsonl), using whichever
embedder and Mistral model the caller's AssistantService already has
loaded -- this module never loads its own model. Running a second copy of
the 7B model alongside the server's would risk exceeding this machine's
single-GPU memory budget (see workspace.md's Task 32 GPU-safety note and
the near-incident logged against this same session) -- call
run_benchmark() from inside the already-running server process (e.g. a
temporary debug endpoint), never as its own standalone script.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import config
import pipeline.classifier as classifier_module
import pipeline.semantic_router as semantic_router_module
from core.assistant_service import AssistantService

GOLD_SET_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "routing_gold_set.jsonl"

_ALL_MODES = ["legacy", "semantic", "hybrid"]


def load_gold_set(path: Path = GOLD_SET_PATH) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _confusion_and_per_class(labels_true: list[str], labels_pred: list[str]) -> tuple[dict, dict]:
    all_labels = sorted(set(labels_true) | set(labels_pred))
    confusion = {t: {p: 0 for p in all_labels} for t in all_labels}
    for t, p in zip(labels_true, labels_pred):
        confusion[t][p] += 1
    per_class = {}
    for label in all_labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion[t].get(label, 0) for t in all_labels if t != label)
        fn = sum(v for k, v in confusion.get(label, {}).items() if k != label)
        per_class[label] = {
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "support": sum(confusion.get(label, {}).values()),
        }
    return confusion, per_class


def run_benchmark(service: AssistantService, modes: list[str] | None = None) -> dict:
    """Runs the full gold set through each requested AUM_ROUTER mode using
    the real routing code paths (service.classify_topic(), same method
    production traffic calls). Returns accuracy, a confusion matrix,
    per-class precision/recall, per-category accuracy, routing latency,
    and the real Mistral-router invocation count per mode -- the number
    that validates or refutes the hybrid design's GPU-cost argument.

    Counts Mistral invocations by wrapping _classify_topic_llm itself
    (not by text-matching routing_path strings) so an invocation that
    returned unparseable output and fell through to a keyword fallback is
    still counted -- the GPU time was still spent even though the label
    was discarded.
    """
    modes = modes or _ALL_MODES
    gold = load_gold_set()

    real_classifier_llm = classifier_module._classify_topic_llm
    real_semantic_llm = semantic_router_module._classify_topic_llm
    call_counter = {"n": 0}

    def counting_llm_call(query, tok, model):
        call_counter["n"] += 1
        return real_classifier_llm(query, tok, model)

    original_router = config.AUM_ROUTER
    results: dict[str, dict] = {}
    classifier_module._classify_topic_llm = counting_llm_call
    semantic_router_module._classify_topic_llm = counting_llm_call
    try:
        for mode in modes:
            config.AUM_ROUTER = mode
            call_counter["n"] = 0
            labels_true: list[str] = []
            labels_pred: list[str] = []
            per_category_correct: dict[str, int] = {}
            per_category_total: dict[str, int] = {}
            latencies_ms: list[float] = []
            mismatches: list[dict] = []

            t_mode_start = time.perf_counter()
            for row in gold:
                t0 = time.perf_counter()
                pred = service.classify_topic(row["query"])
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                labels_true.append(row["expected_topic"])
                labels_pred.append(pred)
                cat = row["category"]
                per_category_total[cat] = per_category_total.get(cat, 0) + 1
                if pred == row["expected_topic"]:
                    per_category_correct[cat] = per_category_correct.get(cat, 0) + 1
                else:
                    mismatches.append({
                        "id": row["id"], "query": row["query"], "category": cat,
                        "expected": row["expected_topic"], "predicted": pred,
                    })
            t_mode_total_s = time.perf_counter() - t_mode_start

            n = len(gold)
            correct = sum(1 for t, p in zip(labels_true, labels_pred) if t == p)
            confusion, per_class = _confusion_and_per_class(labels_true, labels_pred)
            sorted_latencies = sorted(latencies_ms)

            results[mode] = {
                "n": n,
                "accuracy": correct / n,
                "mistral_calls": call_counter["n"],
                "mistral_invocation_rate": call_counter["n"] / n,
                "avg_latency_ms": sum(latencies_ms) / n,
                "p95_latency_ms": sorted_latencies[int(0.95 * n) - 1],
                "max_latency_ms": sorted_latencies[-1],
                "total_wall_time_s": t_mode_total_s,
                "confusion_matrix": confusion,
                "per_class": per_class,
                "per_category_accuracy": {
                    cat: per_category_correct.get(cat, 0) / per_category_total[cat]
                    for cat in per_category_total
                },
                "mismatches": mismatches,
            }
    finally:
        classifier_module._classify_topic_llm = real_classifier_llm
        semantic_router_module._classify_topic_llm = real_semantic_llm
        config.AUM_ROUTER = original_router

    return results
