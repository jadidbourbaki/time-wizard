"""Move the crops between this machine and a Hugging Face dataset.

`timewizard.photos` builds the crops from COCO and OpenImages, which takes
about twenty minutes. `push` uploads the result. `pull` fetches it in about
two. A GPU box therefore starts training straight away.

The dataset carries every crop with its label, source, and split. The split
files under `benchmark/` stay the record of which photograph belongs where.
`photos` rebuilds everything from the source buckets whenever we need to
check the uploaded copy.
"""

from __future__ import annotations

import tyro
from datasets import Dataset, Image, load_dataset
from pydantic import BaseModel
from tqdm import tqdm

from timewizard.photos import CROPS, Split, load_split

REPO_ID = "jadidbourbaki/time-wizard-bench"
SPLITS: tuple[Split, ...] = ("test", "dev", "train")


class CropsConfig(BaseModel):
    repo_id: str = REPO_ID
    private: bool = True


def push(cfg: CropsConfig) -> None:
    for split in SPLITS:
        labels = load_split(split)
        keys = sorted(labels)
        rows = {
            "key": keys,
            "image": [str(CROPS / f"{k}.png") for k in keys],
            "hours": [labels[k].hours for k in keys],
            "minutes": [labels[k].minutes for k in keys],
        }
        dataset = Dataset.from_dict(rows).cast_column("image", Image())
        dataset.push_to_hub(cfg.repo_id, split=split, private=cfg.private)
        print(f"pushed {len(keys)} crops to {cfg.repo_id} as {split}")


def pull(cfg: CropsConfig) -> None:
    CROPS.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        dataset = load_dataset(cfg.repo_id, split=split)
        for row in tqdm(dataset, total=len(dataset), desc=split):
            row["image"].save(CROPS / f"{row['key']}.png")
        print(f"pulled {len(dataset)} crops for {split}")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"push": push, "pull": pull})
