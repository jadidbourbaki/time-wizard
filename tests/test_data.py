from __future__ import annotations

import json

from PIL import Image

from timewizard.data import conversation, with_image
from timewizard.reading import Time


def test_conversation_is_serialisable_and_fillable() -> None:
    messages = conversation(Time(hours=7, minutes=6))
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    json.dumps(messages)

    answer = messages[2]["content"]
    assert isinstance(answer, list) and isinstance(answer[0], dict)
    assert json.loads(answer[0]["text"]) == {"hours": 7, "minutes": 6}

    image = Image.new("RGB", (8, 8))
    content = with_image(messages, image)[1]["content"]
    assert isinstance(content, list) and isinstance(content[0], dict)
    assert content[0]["image"] is image
