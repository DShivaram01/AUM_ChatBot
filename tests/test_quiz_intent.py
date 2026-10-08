"""Task 53 Phase 2: quiz activity-intent detection. Positive/negative
examples are taken directly from
AUM_CHATBOT_INTELLIGENT_QUIZ_ROUTING_AND_RELIABILITY_PLAN_2026-10-08.md
Sections 3, 6, and 9, so this suite doubles as that plan's own acceptance
tests for the detector it specified."""

from pipeline.quiz_intent import detect_quiz_intent, resolve_quiz_source


def test_plan_example_create_5_mcqs_on_computational_biology():
    result = detect_quiz_intent("Create 5 MCQs on computational biology")
    assert result.activity == "quiz_create"
    assert result.count == 5
    assert result.topic == "computational biology"


def test_plan_example_provide_5_mcqs_phrasing():
    # Section 9, functional case 2's exact phrasing.
    result = detect_quiz_intent("provide 5 MCQs on computational biology")
    assert result.activity == "quiz_create"
    assert result.count == 5
    assert result.topic == "computational biology"


def test_plan_example_quiz_me_on_this_pdf():
    result = detect_quiz_intent("Quiz me on this uploaded PDF", has_attached_document=True)
    assert result.activity == "quiz_create"
    assert "pdf" in result.topic.lower()


def test_plan_example_generate_10_questions_about_housing_rules():
    result = detect_quiz_intent("Generate 10 questions about housing rules")
    assert result.activity == "quiz_create"
    assert result.count == 10
    assert result.topic == "housing rules"


def test_plan_negative_example_when_is_my_quiz():
    assert detect_quiz_intent("What time is my quiz?").activity == "chat"


def test_plan_negative_example_what_is_mcq():
    assert detect_quiz_intent("what is MCQ").activity == "chat"


def test_plan_negative_example_how_do_i_create_a_quiz_app():
    assert detect_quiz_intent("how do I create a quiz app").activity == "chat"


def test_plan_negative_example_quiz_schedule():
    assert detect_quiz_intent("quiz schedule").activity == "chat"


def test_plan_negative_example_summarize_my_quiz_results():
    assert detect_quiz_intent("summarize my quiz results").activity == "chat"


def test_plan_negative_example_what_is_computational_biology():
    assert detect_quiz_intent("What is computational biology?").activity == "chat"


def test_count_is_clamped_to_1_through_20():
    assert detect_quiz_intent("generate 99 MCQs on biology").count == 20
    assert detect_quiz_intent("generate 0 MCQs on biology").count == 1


def test_bare_questions_without_a_count_does_not_trigger():
    # "questions" alone is too generic (ordinary chat says "I have
    # questions about X" constantly) -- only counts with an explicit
    # count, or with an unambiguous qualifier like "quiz"/"mcq"/"practice".
    assert detect_quiz_intent("generate questions about biology").activity == "chat"


def test_empty_or_bare_quiz_request_without_document_does_not_trigger():
    assert detect_quiz_intent("quiz me").activity == "chat"
    assert detect_quiz_intent("").activity == "chat"


def test_bare_quiz_request_with_attached_document_gets_a_generic_topic():
    result = detect_quiz_intent("quiz me", has_attached_document=True)
    assert result.activity == "quiz_create"
    assert result.topic


# ---------- resolve_quiz_source ----------

def test_resolve_quiz_source_attached_document_always_wins():
    mode, err = resolve_quiz_source(
        "guest policy", has_attached_document=True, open_ended_enabled=False,
    )
    assert mode == "document" and err is None


def test_resolve_quiz_source_infers_housing_from_topic_hint_words():
    mode, err = resolve_quiz_source(
        "overnight guest policy", has_attached_document=False, open_ended_enabled=False,
    )
    assert mode == "housing" and err is None


def test_resolve_quiz_source_falls_back_to_pretrained_only_if_open_ended():
    mode, err = resolve_quiz_source(
        "computational biology", has_attached_document=False, open_ended_enabled=True,
    )
    assert mode == "pretrained" and err is None


def test_resolve_quiz_source_asks_rather_than_guesses_without_a_source():
    mode, err = resolve_quiz_source(
        "computational biology", has_attached_document=False, open_ended_enabled=False,
    )
    assert mode is None
    assert err and "Open-ended" in err
