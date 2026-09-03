from __future__ import annotations

import torch

from timewizard.train import assistant_mask

HEADER = [6, 64015, 708]
END = 7
SYSTEM_TURN = [1, 6, 24131, 708, 560, 7, 708]
USER_TURN = [6, 6423, 708, 396, 396, 558, 7, 708]
ANSWER = [38685, 574, 602]
PADDING = [0, 0]


def test_assistant_mask_covers_the_answer_and_its_end_token() -> None:
    ids = torch.tensor([SYSTEM_TURN + USER_TURN + HEADER + ANSWER + [END] + PADDING])
    assert ids[assistant_mask(ids, HEADER, END)].tolist() == ANSWER + [END]


def test_assistant_mask_without_an_assistant_turn_is_empty() -> None:
    ids = torch.tensor([SYSTEM_TURN + USER_TURN])
    assert not assistant_mask(ids, HEADER, END).any()
