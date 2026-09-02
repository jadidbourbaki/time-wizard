"""LoRA fine-tune LFM2.5-VL-450M to read clocks.

Follows Liquid's TRL recipe: AutoModelForImageTextToText, a collator that
runs the processor's chat template over whole conversations, SFTTrainer with
dataset preparation skipped. Loss is computed on assistant tokens only.
Training data is the photo train split, SynClock renders, or both. Requires
a CUDA GPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
import tyro
from datasets import Dataset, Features, Image, Value, concatenate_datasets
from peft import LoraConfig
from PIL import Image as PILImage
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

from timewizard import BASE_MODEL
from timewizard.data import conversation, render, with_image
from timewizard.photos import CROPS, Split, load_split
from timewizard.reading import Time

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear", "gate_proj", "up_proj", "down_proj"]
FEATURES = Features({"messages": Value("string"), "image": Image()})

Row = dict[str, Any]
Collator = Callable[[list[Row]], dict[str, torch.Tensor]]


class TrainConfig(BaseModel):
    out: Path
    """Output directory for checkpoints, the adapter, and config.json."""
    photos: bool = True
    """Train on the photo train split."""
    rendered: int = Field(0, ge=0)
    """Number of SynClock renders to add to training."""
    epochs: float = Field(3.0, gt=0)
    seed: int = 0
    image_size: int = Field(448, ge=64)
    batch_size: int = Field(8, ge=1)
    grad_accum: int = Field(4, ge=1)
    lr: float = Field(2e-4, gt=0)
    lora_rank: int = Field(16, ge=1)
    workers: int = Field(8, ge=1)
    model: str = BASE_MODEL


def rendered_rows(indices: list[int], n: int, seed: int, image_size: int) -> Iterator[Row]:
    for i in indices:
        image, time = render(i, n, seed, image_size)
        yield {"messages": json.dumps(conversation(time)), "image": image}


def photo_rows(keys: list[str], labels: dict[str, Time]) -> Iterator[Row]:
    for key in keys:
        with PILImage.open(CROPS / f"{key}.png") as image:
            yield {"messages": json.dumps(conversation(labels[key])), "image": image.convert("RGB")}


def rendered_dataset(n: int, seed: int, image_size: int, workers: int) -> Dataset:
    # `datasets` shards list-valued gen_kwargs across num_proc workers.
    return Dataset.from_generator(
        rendered_rows,
        features=FEATURES,
        gen_kwargs={"indices": list(range(n)), "n": n, "seed": seed, "image_size": image_size},
        num_proc=workers,
    )


def photo_dataset(split: Split, workers: int) -> Dataset:
    labels = load_split(split)
    return Dataset.from_generator(
        photo_rows, features=FEATURES, gen_kwargs={"keys": sorted(labels), "labels": labels}, num_proc=workers
    )


def training_sets(cfg: TrainConfig) -> tuple[Dataset, Dataset]:
    parts = []
    if cfg.photos:
        parts.append(photo_dataset("train", cfg.workers))
    if cfg.rendered:
        parts.append(rendered_dataset(cfg.rendered, cfg.seed, cfg.image_size, cfg.workers))
    if not parts:
        raise SystemExit("nothing to train on: set --photos or --rendered")
    return concatenate_datasets(parts).shuffle(seed=cfg.seed), photo_dataset("dev", cfg.workers)


def assistant_mask(input_ids: torch.Tensor, header: list[int], end_id: int) -> torch.Tensor:
    """True on the tokens of every assistant turn, from the first token after the
    role header through the turn's end token. TRL's assistant_only_loss rejects
    vision datasets, and the tokenizer's own mask is misaligned once the
    processor expands the image placeholder."""
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

    processor = AutoProcessor.from_pretrained(cfg.model, max_image_tokens=256)
    model = AutoModelForImageTextToText.from_pretrained(cfg.model, dtype=torch.bfloat16)
    peft = LoraConfig(
        r=cfg.lora_rank, lora_alpha=2 * cfg.lora_rank, lora_dropout=0.05, bias="none", target_modules=LORA_TARGETS
    )
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        dataloader_num_workers=cfg.workers,
        max_length=None,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="none",
        seed=cfg.seed,
    )
    train_set, dev_set = training_sets(cfg)
    trainer = SFTTrainer(
        model=model,
        args=args,
        data_collator=make_collator(processor),
        train_dataset=train_set,
        eval_dataset=dev_set,
        processing_class=processor,
        peft_config=peft,
    )
    trainer.train()
    trainer.save_model(str(cfg.out / "adapter"))
    processor.save_pretrained(str(cfg.out / "adapter"))


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
