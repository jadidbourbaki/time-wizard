"""Training conversations: one user turn with the image and the prompt, one
assistant turn with the time as JSON."""

from __future__ import annotations

import copy

from PIL import Image

from timewizard.reading import PROMPT, SYSTEM, Time

Message = dict[str, object]


def conversation(time: Time) -> list[Message]:
    """Messages with an image placeholder, so they stay JSON-serialisable.
    `with_image` fills the placeholder."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": time.model_dump_json()}]},
    ]


def with_image(messages: list[Message], image: Image.Image) -> list[Message]:
    out = copy.deepcopy(messages)
    content = out[1]["content"]
    assert isinstance(content, list)
    content[0] = {"type": "image", "image": image}
    return out
