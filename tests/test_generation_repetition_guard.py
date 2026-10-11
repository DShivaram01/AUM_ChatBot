"""Task 51: anti-repetition must not stop the model copying facts from its
own prompt (the retrieved context). See GeneratedOnlyRepetitionGuard."""

import torch
from transformers import NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor

from pipeline.answer import GeneratedOnlyRepetitionGuard

VOCAB = 40
# prompt contains "dues are $ 3 5 per" as ids 7 8 9 3 5 6; the answer is
# copying the same span and is about to emit "5".
PROMPT = [1, 7, 8, 9, 3, 5, 6, 2]
GENERATED = [7, 8, 9, 3]


def _ids(tokens):
    return torch.tensor([tokens])


def test_old_prompt_wide_settings_ban_copying_a_fact_from_context():
    # Documents the bug: HF's processors look at the whole input_ids.
    scores = torch.zeros(1, VOCAB)
    scores = NoRepeatNGramLogitsProcessor(4)(_ids(PROMPT + GENERATED), scores)
    assert scores[0, 5] == -float("inf")
    scores = RepetitionPenaltyLogitsProcessor(1.1)(_ids(PROMPT + GENERATED), torch.ones(1, VOCAB))
    assert scores[0, 5] < 1.0


def test_guard_allows_copying_from_prompt():
    guard = GeneratedOnlyRepetitionGuard(prompt_len=len(PROMPT))
    scores = guard(_ids(PROMPT + GENERATED), torch.ones(1, VOCAB))
    assert scores[0, 5] == 1.0
    assert guard(_ids(PROMPT), torch.ones(1, VOCAB)).eq(1.0).all()  # nothing generated yet


def test_guard_still_blocks_loops_inside_the_answer():
    guard = GeneratedOnlyRepetitionGuard(prompt_len=len(PROMPT))
    looping = [11, 12, 13, 14, 11, 12, 13]  # about to repeat the 4-gram 11 12 13 14
    scores = guard(_ids(PROMPT + looping), torch.ones(1, VOCAB))
    assert scores[0, 14] == -float("inf")
    assert torch.isclose(scores[0, 11], torch.tensor(1.0 / 1.1))  # already generated -> penalised
    assert scores[0, 15] == 1.0        # never generated -> untouched


def test_guard_penalty_handles_negative_logits_and_batches():
    guard = GeneratedOnlyRepetitionGuard(prompt_len=2)
    ids = torch.tensor([[1, 2, 5], [1, 2, 6]])
    scores = guard(ids, -torch.ones(2, VOCAB))
    assert torch.isclose(scores[0, 5], torch.tensor(-1.1)) and scores[0, 6] == -1.0
    assert torch.isclose(scores[1, 6], torch.tensor(-1.1)) and scores[1, 5] == -1.0


def test_guard_bans_reproducing_an_avoided_sequence_but_not_the_prompt():
    # quiz: the previous question (ids 11 12 13 14) must not be re-written,
    # while copying from the context in the prompt stays allowed.
    previous = list(range(1, 20))  # a 19-token earlier question
    guard = GeneratedOnlyRepetitionGuard(prompt_len=len(PROMPT), banned_sequences=[previous])
    scores = guard(_ids(PROMPT + previous[:3]), torch.ones(1, VOCAB))
    assert scores[0, 4] == -float("inf")  # re-writing it is blocked from the 4th token
    scores = guard(_ids(PROMPT + previous[:2]), torch.ones(1, VOCAB))
    assert scores[0, 3] != -float("inf")  # shorter shared spans are fine
    scores = guard(_ids(PROMPT + GENERATED), torch.ones(1, VOCAB))
    assert scores[0, 5] == 1.0
