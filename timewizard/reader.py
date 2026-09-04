"""Read the time from one clock photograph with the fine-tuned model.

`Reader` loads the model once and answers many photographs. The command
line takes one image and prints the time as JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import tyro
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from timewizard.reading import PROMPT, SYSTEM, Time, parse_time

MODEL = "jadidbourbaki/time-wizard"
IMAGE_TOKENS = 256
IMAGE_SIZE = 448
MAX_NEW_TOKENS = 32


class ReadConfig(BaseModel):
    image: tyro.conf.Positional[Path]
    """A photograph cropped to the clock."""
    checkpoint: str = MODEL
    """A local directory or a Hugging Face repo id."""


class Reader:
    def __init__(self, checkpoint: str = MODEL) -> None:
        self.processor = AutoProcessor.from_pretrained(checkpoint, max_image_tokens=IMAGE_TOKENS)
        self.model: Any = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.bfloat16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)

    def reply(self, image: Image.Image) -> str:
        """The model's raw reply for `image`, padded to the square the model was trained on."""
        square = ImageOps.pad(image.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": square}, {"type": "text", "text": PROMPT}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()

    def read(self, image: Image.Image) -> Time | None:
        """The time on the clock, or None when the reply holds no valid time."""
        return parse_time(self.reply(image))


def main(cfg: ReadConfig) -> None:
    with Image.open(cfg.image) as image:
        time = Reader(cfg.checkpoint).read(image)
    if time is None:
        raise SystemExit("no valid time in the reply")
    print(time.model_dump_json())


if __name__ == "__main__":
    main(tyro.cli(ReadConfig))
