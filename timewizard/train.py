"""Fine-tune LFM2.5-VL-450M on the photo train split.

Every weight trains, vision encoder included. A 450M model fits one GPU
with room to spare, and reading hand angles asks the encoder to change.
Loss covers assistant tokens only. Requires a CUDA GPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
import tyro
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

from timewizard import BASE_MODEL
from timewizard.data import conversation, with_image
from timewizard.photos import CROPS, Split, load_split
from timewizard.reading import Time

FEATURES = Features({"messages": Value("string"), "image": Image()})
IMAGE_TOKENS = 256

Row = dict[str, Any]
Collator = Callable[[list[Row]], dict[str, torch.Tensor]]


class TrainConfig(BaseModel):
    out: Path
    """Directory for config.json, checkpoints, and the final model."""
    hub: str | None = "jadidbourbaki/time-wizard"
    """Private Hugging Face repo that receives the final model. None skips the upload."""
    epochs: float = Field(5.0, gt=0)
    seed: int = 0
    batch_size: int = Field(16, ge=1)
    grad_accum: int = Field(2, ge=1)
    lr: float = Field(5e-5, gt=0)
    workers: int = Field(8, ge=1)
    model: str = BASE_MODEL


def photo_rows(keys: list[str], labels: dict[str, Time]) -> Iterator[Row]:
    for key in keys:
        with PILImage.open(CROPS / f"{key}.png") as image:
            yield {"messages": json.dumps(conversation(labels[key])), "image": image.convert("RGB")}


def photo_dataset(split: Split, workers: int) -> Dataset:
    labels = load_split(split)
    return Dataset.from_generator(
        photo_rows, features=FEATURES, gen_kwargs={"keys": sorted(labels), "labels": labels}, num_proc=workers
    )


def assistant_mask(input_ids: torch.Tensor, header: list[int], end_id: int) -> torch.Tensor:
    """True on the tokens of every assistant turn, from the first token after
    the role header through the turn's end token. The spans are found here
    because the processor expands the image placeholder after the tokenizer
    builds its own mask, which shifts every position."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    width = len(header)
    for row, out in zip(input_ids.tolist(), mask, strict=True):
        i = 0
        while i <= len(row) - width:
            if row[i : i + width] == header:
                j = i + width
                while j < len(row) and row[j] != end_id:
                    j += 1
                out[i + width : j + 1] = True
                i = j
            i += 1
    return mask


def make_collator(processor: Any) -> Collator:
    header = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def collate(rows: list[Row]) -> dict[str, torch.Tensor]:
        conversations = [with_image(json.loads(r["messages"]), r["image"].convert("RGB")) for r in rows]
        batch = processor.apply_chat_template(
            conversations, tokenize=True, return_dict=True, return_tensors="pt", padding=True
        )
        supervised = assistant_mask(batch["input_ids"], header, end_id)
        batch["labels"] = batch["input_ids"].masked_fill(~supervised, -100)
        return batch

    return collate


def main(cfg: TrainConfig) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "config.json").write_text(cfg.model_dump_json(indent=2))

    processor = AutoProcessor.from_pretrained(cfg.model, max_image_tokens=IMAGE_TOKENS)
    model: Any = AutoModelForImageTextToText.from_pretrained(cfg.model, dtype=torch.bfloat16)
    args = SFTConfig(
        output_dir=str(cfg.out),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.workers,
        max_length=None,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="none",
        seed=cfg.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        data_collator=make_collator(processor),
        train_dataset=photo_dataset("train", cfg.workers),
        eval_dataset=photo_dataset("dev", cfg.workers),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(cfg.out / "model"))
    processor.save_pretrained(str(cfg.out / "model"))
    if cfg.hub:
        model.push_to_hub(cfg.hub, private=True)
        processor.push_to_hub(cfg.hub, private=True)


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
