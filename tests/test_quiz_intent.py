"""Task 53 Phase 2: quiz activity-intent detection. Positive/negative
examples are taken directly from
AUM_CHATBOT_INTELLIGENT_QUIZ_ROUTING_AND_RELIABILITY_PLAN_2026-10-08.md
Sections 3, 6, and 9, so this suite doubles as that plan's own acceptance
tests for the detector it specified."""

from pipeline.quiz_intent import detect_quiz_followup, detect_quiz_intent, resolve_quiz_source


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


# ---------- Task 58: recall fixes, checked on phrasings NOT in activity_gold_set ----------

def test_task58_new_phrasings_trigger_quiz_creation():
    for text, topic in [
        ("I need 10 MCQs on genetics", "genetics"),
        ("Could you make me a short multiple-choice quiz on Rome?", "Rome"),
        ("can you get me some practice questions on calculus", "calculus"),
        ("Generate a 12-question quiz about plate tectonics", "plate tectonics"),
        ("I want a mock exam on organic chemistry", "organic chemistry"),
    ]:
        r = detect_quiz_intent(text)
        assert r.activity == "quiz_create", text
        assert r.topic == topic, (text, r.topic)
    assert detect_quiz_intent("Generate a 12-question quiz about plate tectonics").count == 12


def test_task58_new_verbs_do_not_hijack_ordinary_chat():
    for text in [
        "I need the quiz answers from yesterday", "I want to know when the quiz is",
        "Can I get my quiz grade?", "I need help with my quiz schedule",
        "write a unit test for this function", "I need a practice room in the library",
        "Make the next quiz easier", "get me the dorm guest policy",
    ]:
        assert detect_quiz_intent(text).activity == "chat", text


def test_task58_followup_ordinals_last_and_answer_wording():
    assert detect_quiz_followup("Can you explain the second question?") == {"kind": "explain", "question_number": 2}
    assert detect_quiz_followup("Show me the answer to the last question") == {"kind": "explain", "question_number": -1}
    assert detect_quiz_followup("What was wrong with my answer on question 4?")["question_number"] == 4
    assert detect_quiz_followup("Generate another 5 questions") == {"kind": "regenerate"}
    assert detect_quiz_followup("More questions like these please") == {"kind": "regenerate"}


# ---------- detect_quiz_followup ----------

def test_explain_question_n_both_word_orders():
    assert detect_quiz_followup("explain question 3") == {"kind": "explain", "question_number": 3}
    assert detect_quiz_followup("why is the answer to question 2 correct?") == {
        "kind": "explain", "question_number": 2,
    }


def test_explain_without_a_question_number_does_not_match():
    assert detect_quiz_followup("explain that in more detail") is None


def test_question_number_without_explain_trigger_does_not_match():
    assert detect_quiz_followup("question 3 was easy") is None


def test_regenerate_phrasings():
    for phrase in [
        "try again", "regenerate", "make another one", "give me another quiz",
        "another quiz on the same topic", "redo this quiz",
    ]:
        assert detect_quiz_followup(phrase) == {"kind": "regenerate"}, phrase


def test_ordinary_messages_do_not_match_followup():
    assert detect_quiz_followup("What is the housing guest policy?") is None
    assert detect_quiz_followup("") is None
