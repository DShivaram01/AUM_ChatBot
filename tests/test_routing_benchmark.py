"""Task 48: routing benchmark is safe by construction -- no global config
mutation, no module monkey-patching, Mistral calls counted per arm."""

import config
import pipeline.classifier as classifier_module
import pipeline.semantic_router as semantic_router_module
from core.runtime_manager import RuntimeManager
from pipeline.routing_benchmark import run_benchmark, score


class FakeModel:
    device = "cpu"

    def generate(self, **_kwargs):
        return "out"


ROWS = [
    {"id": "a", "query": "q1", "expected_topic": "cos", "category": "x"},
    {"id": "b", "query": "q2", "expected_topic": "housing", "category": "x"},
    {"id": "c", "query": "q3", "expected_topic": "housing", "category": "y"},
]


def test_run_benchmark_touches_no_global_state_and_counts_calls_per_arm():
    before = (config.AUM_ROUTER, classifier_module._classify_topic_llm,
              semantic_router_module._classify_topic_llm)
    seen_mid_run = []

    def calls_model_twice(query, model):
        seen_mid_run.append(config.AUM_ROUTER)  # what a concurrent live request would see
        model.generate()
        model.generate()
        return "housing"

    results = run_benchmark(
        RuntimeManager(llm_model=FakeModel()),
        routers={"always_housing": calls_model_twice, "no_llm": lambda q, m: "cos"},
        rows=ROWS,
    )
    after = (config.AUM_ROUTER, classifier_module._classify_topic_llm,
             semantic_router_module._classify_topic_llm)
    assert before == after
    assert set(seen_mid_run) == {before[0]}  # never flipped during the run
    assert results["always_housing"]["mistral_calls"] == 6  # counted per arm, not shared
    assert results["no_llm"]["mistral_calls"] == 0
    assert results["always_housing"]["accuracy"] == 2 / 3


def test_holdout_split_has_no_prototype_leakage():
    from pipeline.routing_benchmark import load_gold_set
    from pipeline.semantic_router import ROUTING_EXAMPLES
    holdout, calib = load_gold_set(split="holdout"), load_gold_set(split="calibration")
    assert holdout and calib and len(holdout) + len(calib) == len(load_gold_set())
    assert all(r["proto_max_sim"] < 0.85 for r in holdout)
    protos = {e.lower().strip() for exs in ROUTING_EXAMPLES.values() for e in exs}
    assert not any(r["query"].lower().strip() in protos for r in holdout)


def test_score_macro_f1_exposes_a_router_that_ignores_a_class():
    truth = ["cos"] * 8 + ["out_of_scope"] * 2
    never_oos = ["cos"] * 10
    s = score(truth, never_oos)
    assert s["accuracy"] == 0.8               # looks fine...
    assert s["per_class"]["out_of_scope"]["recall"] == 0.0
    assert s["macro_f1"] < 0.5                # ...macro-F1 does not
    assert s["confusion_matrix"]["out_of_scope"]["cos"] == 2
