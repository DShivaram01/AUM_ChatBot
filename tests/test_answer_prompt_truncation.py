"""Task 52: generate_streaming()'s prompt-length safety net must never
delete the closing "[/INST]" tag -- found via a real quiz failure where
grounded quiz prompts routinely exceeded the old limit and silently lost
their instruction closer, leaving the model nothing to respond to."""

from pipeline.answer import _truncate_prompt


def test_short_prompt_is_returned_unchanged():
    prompt = "<s>[INST] short [/INST]"
    assert _truncate_prompt(prompt, max_chars=3200) == prompt


def test_long_prompt_keeps_closing_inst_tag_instead_of_losing_it():
    body = "x" * 5000
    prompt = f"<s>[INST] instructions\n<context>\n{body}\n</context> [/INST]"
    truncated = _truncate_prompt(prompt, max_chars=3200)
    assert len(truncated) == 3200
    assert truncated.endswith("[/INST]")
    # The naive prompt[:3200] slice (the old behavior) would NOT end here --
    # confirm this test actually distinguishes the fix from the bug.
    assert prompt[:3200] != truncated
    assert not prompt[:3200].endswith("[/INST]")


def test_prompt_without_closing_tag_falls_back_to_plain_slice():
    prompt = "no instruction tag here, just " + ("y" * 5000)
    truncated = _truncate_prompt(prompt, max_chars=3200)
    assert truncated == prompt[:3200]
