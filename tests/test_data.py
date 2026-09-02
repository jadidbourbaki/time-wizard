from __future__ import annotations

import json
import random

import pytest

from timewizard.clocks import sample
from timewizard.data import conversation, render, with_image
from timewizard.reading import Time


@pytest.mark.parametrize("seed", range(5))
def test_rendered_labels_are_valid_times(seed: int) -> None:
    image, time = sample(random.Random(seed))
    assert image.size == (448, 448)
    assert 1 <= time.hours <= 12 and 0 <= time.minutes <= 59


def test_render_is_deterministic_in_index_and_seed() -> None:
    a, b = render(3, 100, 0, 224), render(3, 100, 0, 224)
    assert a[1] == b[1] and a[0].tobytes() == b[0].tobytes() and a[0].size == (224, 224)
    assert render(3, 100, 1, 224)[1] != a[1] or render(4, 100, 0, 224)[1] != a[1]


def test_conversation_is_serialisable_and_fillable() -> None:
    messages = conversation(Time(hours=7, minutes=6))
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    json.dumps(messages)
    image, _ = render(0, 10, 0, 64)
    content = with_image(messages, image)[1]["content"]
    assert isinstance(content, list) and isinstance(content[0], dict) and content[0]["image"] is image
    answer = messages[2]["content"]
    assert isinstance(answer, list) and isinstance(answer[0], dict)
    assert json.loads(answer[0]["text"]) == {"hours": 7, "minutes": 6}
