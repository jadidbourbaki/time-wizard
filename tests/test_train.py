from __future__ import annotations

import torch

from timewizard.train import assistant_mask

HEADER, END, PAD = [6, 64015, 708], 7, 0


def test_assistant_mask_covers_answers_and_end_tokens_only() -> None:
    # system turn, user turn with two image tokens, assistant "{a}", padding
    ids = torch.tensor(
        [[1, 6, 24131, 708, 560, 7, 708, 6, 6423, 708, 396, 396, 558, 7, 708] + HEADER + [38685, 574, 602, 7, PAD, PAD]]
    )
    assert ids[assistant_mask(ids, HEADER, END)].tolist() == [38685, 574, 602, 7]


def test_assistant_mask_without_assistant_turn_is_empty() -> None:
    assert not assistant_mask(torch.tensor([[1, 6, 6423, 708, 558, 7, 708]]), HEADER, END).any()
