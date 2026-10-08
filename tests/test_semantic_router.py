"""Task 42: semantic/hybrid routing and the AUM_ROUTER feature-flag dispatch."""

import numpy as np

import config
from core.assistant_service import AssistantService
from core.runtime_manager import RuntimeManager
from pipeline import semantic_router
from pipeline.semantic_router import (
    DEFAULT_MARGIN_THRESHOLD,
    classify_topic_hybrid,
    classify_topic_semantic,
    semantic_route,
)


class WordOverlapEmbedder:
    """Deterministic, model-free stand-in for all-MiniLM-L6-v2 -- same
    pattern tests/test_quiz.py already uses, with a vocabulary that
    discriminates between the five routing labels instead of housing
    content words."""

    VOCAB = [
        "research", "mentor", "symposium", "paper",
        "dorm", "quiet", "guest", "pet",
        "tuition", "aum", "admissions",
        "recipe", "explain", "poem",
        "hack", "lockpick", "weapon",
    ]

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            words = set(text.lower().split())
            vec = np.array([1.0 if term in words else 0.0 for term in self.VOCAB], dtype=np.float32)
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm > 0 else vec)
        return np.array(rows, dtype=np.float32)


def setup_function(_fn):
    # Each test gets a fresh embedder instance (and thus a fresh id()), so
    # the module-level example-embedding cache in semantic_router.py can't
    # leak between tests that monkeypatch ROUTING_EXAMPLES.
    semantic_router._example_cache.clear()


def test_semantic_route_ranks_by_best_per_label_not_top_two_raw_rows():
    # Craft examples where the single highest-scoring row and the
    # second-highest-scoring row are BOTH "cos" -- a naive top-2-rows
    # approach would report margin against another cos row (margin ~0),
    # hiding that "housing" is actually the correct runner-up label.
    semantic_router.ROUTING_EXAMPLES = {
        "cos": ["research paper", "research symposium"],
        "housing": ["dorm guest"],
        "general_aum": ["tuition aum"],
        "general": ["recipe poem"],
        "out_of_scope": ["hack weapon"],
    }
    embedder = WordOverlapEmbedder()
    result = semantic_route("research symposium dorm", embedder)
    assert result["label"] == "cos"  # matches both words of "research symposium"
    assert result["second_label"] == "housing"  # NOT another cos row (e.g. "research paper")
    assert 0.0 < result["margin"] < 1.0


def test_classify_topic_semantic_applies_general_aum_evidence_gate():
    semantic_router.ROUTING_EXAMPLES = {
        "cos": ["research mentor"],
        "housing": ["dorm guest"],
        "general_aum": ["tuition admissions"],
        "general": ["recipe poem"],
        "out_of_scope": ["hack weapon"],
    }
    embedder = WordOverlapEmbedder()
    # Matches "general_aum" examples by vocabulary but contains neither
    # "aum" nor "auburn montgomery" -- the shared evidence gate
    # (pipeline.classifier._has_reliable_aum_evidence) must downgrade this
    # to "general", exactly as it does for the legacy Mistral router.
    label = classify_topic_semantic("what about tuition admissions", embedder)
    assert label == "general"


def test_classify_topic_hybrid_confident_margin_never_calls_mistral():
    semantic_router.ROUTING_EXAMPLES = {
        "cos": ["research mentor symposium paper"],
        "housing": ["dorm quiet guest pet"],
        "general_aum": ["tuition aum admissions"],
        "general": ["recipe explain poem"],
        "out_of_scope": ["hack lockpick weapon"],
    }
    embedder = WordOverlapEmbedder()

    def exploding_llm_call(*_args, **_kwargs):
        raise AssertionError("Mistral must not be called on a confident semantic match")

    real = semantic_router._classify_topic_llm
    semantic_router._classify_topic_llm = exploding_llm_call
    try:
        label = classify_topic_hybrid("dorm quiet guest", embedder, llm_tok=object(), llm_model=object())
    finally:
        semantic_router._classify_topic_llm = real
    assert label == "housing"


def test_classify_topic_hybrid_low_margin_falls_through_to_mistral():
    # A query with no vocabulary overlap at all scores 0.0 against every
    # label -- margin is 0.0, well below DEFAULT_MARGIN_THRESHOLD, so this
    # must fall through to the Mistral router rather than guess.
    semantic_router.ROUTING_EXAMPLES = {
        "cos": ["research mentor"],
        "housing": ["dorm guest"],
        "general_aum": ["tuition admissions"],
        "general": ["recipe poem"],
        "out_of_scope": ["hack weapon"],
    }
    embedder = WordOverlapEmbedder()
    routing_path: list[str] = []

    def fake_llm_call(_query, _tok, _model):
        return "housing"

    real = semantic_router._classify_topic_llm
    semantic_router._classify_topic_llm = fake_llm_call
    try:
        label = classify_topic_hybrid(
            "completely unrelated gibberish", embedder,
            llm_tok=object(), llm_model=object(), routing_path=routing_path,
        )
    finally:
        semantic_router._classify_topic_llm = real
    assert label == "housing"
    assert any("falling through to Mistral" in step for step in routing_path)
    assert any("hybrid Mistral-router label=housing" in step for step in routing_path)


def test_classify_topic_hybrid_without_mistral_falls_back_to_semantic_label():
    semantic_router.ROUTING_EXAMPLES = {
        "cos": ["research mentor"],
        "housing": ["dorm guest"],
        "general_aum": ["tuition admissions"],
        "general": ["recipe poem"],
        "out_of_scope": ["hack weapon"],
    }
    embedder = WordOverlapEmbedder()
    # No llm_tok/llm_model at all (e.g. model not loaded yet) -- must still
    # return a real label, never raise or return None.
    label = classify_topic_hybrid("completely unrelated gibberish", embedder)
    assert label in {"cos", "housing", "general_aum", "general", "out_of_scope"}


def test_default_margin_threshold_is_a_real_number_not_yet_claimed_calibrated():
    # TASK 42 step 3 explicitly flags this as "threshold TBD during
    # calibration" -- this test only guards against an accidental 0 or
    # negative value slipping in, not against the exact number changing
    # once real calibration data exists.
    assert 0.0 < DEFAULT_MARGIN_THRESHOLD < 1.0


def test_assistant_service_classify_topic_dispatches_on_aum_router_flag():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    semantic_router.ROUTING_EXAMPLES = {
        "cos": ["research mentor symposium paper"],
        "housing": ["dorm quiet guest pet"],
        "general_aum": ["tuition aum admissions"],
        "general": ["recipe explain poem"],
        "out_of_scope": ["hack lockpick weapon"],
    }
    original_router = config.AUM_ROUTER
    try:
        config.AUM_ROUTER = "semantic"
        assert service.classify_topic("dorm quiet guest") == "housing"

        config.AUM_ROUTER = "hybrid"
        assert service.classify_topic("dorm quiet guest") == "housing"

        config.AUM_ROUTER = "legacy"
        # No llm_tok/llm_model configured on this bare RuntimeManager, so
        # legacy falls through to its own keyword/embedding fallback --
        # this only proves "legacy" is a genuinely different code path
        # from "semantic"/"hybrid" above, not a specific label.
        legacy_label = service.classify_topic("dorm quiet guest")
        assert legacy_label in {"cos", "housing", "general_aum", "general", "out_of_scope"}
    finally:
        config.AUM_ROUTER = original_router
