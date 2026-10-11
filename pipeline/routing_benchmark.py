"""Routing benchmark against a labeled gold set (TASK 42), safe by
construction (TASK 48).

Never loads its own model: it uses whatever embedder/Mistral the caller's
RuntimeManager already holds, so run it inside the already-running server
process -- a standalone script would load a second 7B copy onto the single
GPU (workspace.md Task 32).

TASK 48: the first version set config.AUM_ROUTER and monkey-patched
_classify_topic_llm on two modules for the duration of a run, so live
requests served mid-run could see the wrong router. Now each router is an
injected callable, called directly, and Mistral calls are counted through a
per-arm wrapper around the model object -- no global state is touched.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from pipeline.classifier import classify_topic
from pipeline.semantic_router import classify_topic_hybrid, classify_topic_semantic

GOLD_SET_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "routing_gold_set.jsonl"

# A router arm: (query, llm_model_or_None) -> one of the five topic labels.
Router = Callable[[str, object], str]


class _CountingModel:
    """Counts generate() calls on one arm's model without patching modules."""

    def __init__(self, model):
        self._model = model
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return self._model.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def load_gold_set(path: Path = GOLD_SET_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def production_routers(runtime) -> dict[str, Router]:
    """The three shipped architectures, called directly -- not through
    AssistantService.classify_topic(), which dispatches on global config."""
    r = runtime
    return {
        "legacy": lambda q, m: classify_topic(
            q, r.embedder, r.cos_index, r.housing_index, r.housing_ok, r.llm_tok, m,
        ),
        "semantic": lambda q, m: classify_topic_semantic(q, r.embedder),
        "hybrid": lambda q, m: classify_topic_hybrid(q, r.embedder, r.llm_tok, m),
    }


def _percentile(sorted_vals: list[float], pct: float) -> float:
    # nearest-rank; safe for small n (the old int(0.95*n)-1 indexed -1 at n<2)
    return sorted_vals[max(0, min(len(sorted_vals) - 1, round(pct * len(sorted_vals)) - 1))]


def score(labels_true: list[str], labels_pred: list[str]) -> dict:
    """Accuracy, macro-F1, per-class P/R/F1 and a confusion matrix. Macro-F1
    matters because the five classes are imbalanced -- accuracy alone hides a
    router that never predicts a small class like out_of_scope."""
    labels = sorted(set(labels_true) | set(labels_pred))
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(labels_true, labels_pred):
        confusion[t][p] += 1
    per_class = {}
    for lab in labels:
        tp = confusion[lab][lab]
        fp = sum(confusion[t][lab] for t in labels if t != lab)
        fn = sum(confusion[lab][p] for p in labels if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    gold_labels = sorted(set(labels_true))
    n = len(labels_true)
    return {
        "n": n,
        "accuracy": sum(t == p for t, p in zip(labels_true, labels_pred)) / n if n else 0.0,
        "macro_f1": sum(per_class[lab]["f1"] for lab in gold_labels) / len(gold_labels) if gold_labels else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def run_benchmark(
    runtime, routers: dict[str, Router] | None = None, rows: list[dict] | None = None,
) -> dict:
    """Run every row through every router arm. Returns per-arm metrics
    (score() plus latency P50/P95, real Mistral generate() count, per-category
    accuracy and mismatches)."""
    routers = routers or production_routers(runtime)
    rows = rows if rows is not None else load_gold_set()
    results: dict[str, dict] = {}
    for name, route in routers.items():
        model = _CountingModel(runtime.llm_model) if runtime.llm_model is not None else None
        preds, latencies = [], []
        start = time.perf_counter()
        for row in rows:
            t0 = time.perf_counter()
            preds.append(route(row["query"], model))
            latencies.append((time.perf_counter() - t0) * 1000.0)
        wall = time.perf_counter() - start

        truth = [row["expected_topic"] for row in rows]
        res = score(truth, preds)
        lat = sorted(latencies)
        cats: dict[str, list[int]] = {}
        for row, pred in zip(rows, preds):
            cats.setdefault(row.get("category", "uncategorized"), []).append(pred == row["expected_topic"])
        calls = model.calls if model else 0
        res.update({
            "mistral_calls": calls,
            "mistral_invocation_rate": calls / len(rows) if rows else 0.0,
            "p50_latency_ms": _percentile(lat, 0.50) if lat else 0.0,
            "p95_latency_ms": _percentile(lat, 0.95) if lat else 0.0,
            "avg_latency_ms": sum(lat) / len(lat) if lat else 0.0,
            "total_wall_time_s": wall,
            "per_category_accuracy": {c: sum(v) / len(v) for c, v in cats.items()},
            "mismatches": [
                {"id": row.get("id"), "query": row["query"], "category": row.get("category"),
                 "expected": row["expected_topic"], "predicted": pred}
                for row, pred in zip(rows, preds) if pred != row["expected_topic"]
            ],
        })
        results[name] = res
    return results
