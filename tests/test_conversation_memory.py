"""Task 54: conversation memory. Before this, pipeline/memory.py's
format_history() was a documented no-op stub -- there was genuinely no
conversation memory anywhere in this app. Scoped to Housing, Document QA,
and General; COS is deliberately excluded (see format_history's own
docstring for why)."""

from pipeline.memory import format_history


def test_empty_history_returns_empty_string():
    assert format_history([]) == ""
    assert format_history(None) == ""


def test_formats_question_answer_pairs_oldest_first():
    history = [
        {"question": "What is the guest policy?", "answer": "Guests may stay up to 3 nights."},
        {"question": "What about pets?", "answer": "Only fish under 10 gallons are allowed."},
    ]
    block = format_history(history)
    assert "Earlier in this conversation:" in block
    assert block.index("guest policy") < block.index("pets")
    assert "Guests may stay up to 3 nights" in block
    assert "fish under 10 gallons" in block


def test_keeps_only_the_last_max_turns():
    history = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
    block = format_history(history, max_turns=4)
    assert "Q0" not in block and "Q5" not in block
    assert "Q6" in block and "Q9" in block


def test_truncates_oldest_turns_first_when_over_max_chars():
    history = [
        {"question": "old question", "answer": "x" * 500},
        {"question": "recent question", "answer": "y" * 500},
    ]
    block = format_history(history, max_turns=4, max_chars=600)
    assert "recent question" in block
    assert "y" * 500 in block
    assert "old question" not in block


def test_skips_blank_turns():
    history = [{"question": "", "answer": ""}, {"question": "real one", "answer": "a real answer"}]
    block = format_history(history)
    assert "real one" in block
    assert block.count("Student:") == 1
