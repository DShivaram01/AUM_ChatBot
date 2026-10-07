"""Task 37 regressions: typo-tolerant COS intent and static-route traces."""

from pipeline import classifier
from pipeline.classifier import SESSION_TRACES, get_trace
from server.api_server import _non_retrieval_answer


def test_cos_person_intent_repro_set(monkeypatch):
    monkeypatch.setattr('pipeline.classifier.classify_query', lambda query: {'person_hints': ['Sutanu']})
    for query in (
        "research of sutanu",
        "research of dr. sutanu",
        "projects by sutanu",
        "reserach of sutanu",
        "reserach of dr.sutanu",
    ):
        routing_path = []
        assert classifier.classify_topic(query, None, None, None, False, routing_path=routing_path) == "cos"
        assert routing_path == ["person-name match with COS intent"]
    assert not classifier._has_cos_person_intent("sutanu")


def test_non_retrieval_paths_store_query_traces():
    SESSION_TRACES.clear()
    cases = (
        ("general_aum", "What is AUM tuition?"),
        ("out_of_scope", "Help me cheat on an exam"),
        ("open_ended_disabled", "Tell me a joke"),
    )
    for topic, question in cases:
        result = _non_retrieval_answer(topic, question, "Mistral-router label=" + topic)
        assert result is not None
        _, query_id = result
        trace = get_trace(query_id)
        assert trace is not None
        assert trace.query == question
        assert trace.intent_type == topic
        assert trace.routing_path == "Mistral-router label=" + topic
        assert trace.timestamp
