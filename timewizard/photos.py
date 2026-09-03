"""Real clock photos from It's About Time (Yang and Zisserman, 2022).

`pull` fetches the finished crops from a Hugging Face dataset in about two
minutes. Use it to set up a machine.

`build` recreates them from scratch, which takes about twenty minutes. It
reads the labels and detector boxes from the CSVs in the itsabouttime repo
at the pinned commit. It fetches each image from COCO or OpenImages. It
crops around the detector box with the paper's 20 percent margin. It pads
the crop to a square. It drops near duplicates by perceptual hash. It then
writes the splits. `push` uploads that result.

`benchmark/` holds the frozen splits as image ids plus labels. It stays the
record of which photograph belongs where. The pixels live in `data/` under
the source licences.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import random
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import imagehash
import tyro
from datasets import Dataset, load_dataset
from datasets import Image as ImageColumn
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, TypeAdapter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

from timewizard.reading import Time

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "photos"
CROPS = DATA / "crops"
BENCHMARK = REPO / "benchmark"
LABELS_URL = (
    "https://raw.githubusercontent.com/jadidbourbaki/itsabouttime/248523d7269461a43f8588dcede7473c2edd19fd/data/{}.csv"
)

Source = Literal["coco", "openimages"]
Split = Literal["test", "dev", "train"]
CSV_NAMES: dict[Source, str] = {"coco": "coco_final", "openimages": "openimg_final"}
IMAGE_URLS: dict[Source, list[str]] = {
    "coco": [f"http://images.cocodataset.org/{s}2017/{{}}" for s in ("train", "val")],
    "openimages": [f"https://open-images-dataset.s3.amazonaws.com/{s}/{{}}" for s in ("train", "validation", "test")],
}
MARGIN = 0.2
CROP_SIZE = 448
DUPLICATE_BITS = 4
SPLIT_FILE = TypeAdapter(dict[str, Time])
DATASET = "jadidbourbaki/time-wizard-bench"
SPLITS: tuple[Split, ...] = ("test", "dev", "train")


class Photo(BaseModel):
    source: Source
    file_name: str
    bbox: tuple[float, float, float, float]
    time: Time

    @property
    def key(self) -> str:
        return f"{self.source}_{Path(self.file_name).stem}"

    @property
    def path(self) -> Path:
        return CROPS / f"{self.key}.png"

    def urls(self) -> list[str]:
        return [url.format(self.file_name) for url in IMAGE_URLS[self.source]]


class BuildConfig(BaseModel):
    seed: int = 0
    test_n: int = Field(200, ge=1)
    dev_n: int = Field(200, ge=1)
    workers: int = Field(16, ge=1)


class HubConfig(BaseModel):
    dataset: str = DATASET
    private: bool = True


def load_labels() -> list[Photo]:
    photos = []
    for source, name in CSV_NAMES.items():
        path = DATA / f"{name}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(LABELS_URL.format(name), path)
        for row in csv.DictReader(path.open()):
            x, y, w, h = ast.literal_eval(row["bbox_det"])
            time = Time(hours=int(row["hour"]), minutes=int(row["minute"]))
            photos.append(Photo(source=source, file_name=row["file_name"], bbox=(x, y, w, h), time=time))
    return photos


@retry(
    retry=retry_if_exception_type((urllib.error.URLError, TimeoutError, ConnectionError)),
    wait=wait_exponential(min=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def fetch(url: str) -> bytes | None:
    """Bytes at `url`, or None when that bucket does not hold the image."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise


def crop(photo: Photo) -> Photo | None:
    """Write the padded square crop for `photo`, or None when no bucket has it."""
    if photo.path.exists():
        return photo
    for url in photo.urls():
        data = fetch(url)
        if data is None:
            continue
        image = Image.open(io.BytesIO(data)).convert("RGB")
        x, y, w, h = photo.bbox
        mx, my = w * MARGIN, h * MARGIN
        box = (max(0, x - mx), max(0, y - my), min(image.width, x + w + mx), min(image.height, y + h + my))
        ImageOps.pad(image.crop(box), (CROP_SIZE, CROP_SIZE), color=(0, 0, 0)).save(photo.path)
        return photo
    return None


def crop_all(photos: list[Photo], workers: int) -> list[Photo]:
    CROPS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        done = tqdm(pool.map(crop, photos), total=len(photos), desc="crops")
        return [photo for photo in done if photo is not None]


def deduplicate(photos: list[Photo]) -> list[Photo]:
    """Drop crops whose perceptual hash is within DUPLICATE_BITS of an earlier one."""
    seen: list[imagehash.ImageHash] = []
    kept = []
    for photo in photos:
        digest = imagehash.phash(Image.open(photo.path))
        if all(digest - other > DUPLICATE_BITS for other in seen):
            seen.append(digest)
            kept.append(photo)
    return kept


def split(photos: list[Photo], seed: int, test_n: int, dev_n: int) -> dict[Split, list[Photo]]:
    """Seeded split, stratified by source in proportion to source size."""
    rng = random.Random(seed)
    out: dict[Split, list[Photo]] = {"test": [], "dev": [], "train": []}
    for source in CSV_NAMES:
        group = sorted((p for p in photos if p.source == source), key=lambda p: p.key)
        rng.shuffle(group)
        test = round(test_n * len(group) / len(photos))
        dev = round(dev_n * len(group) / len(photos))
        out["test"] += group[:test]
        out["dev"] += group[test : test + dev]
        out["train"] += group[test + dev :]
    return out


def load_split(name: Split) -> dict[str, Time]:
    """Frozen split as key to label. Crops are at CROPS / f"{key}.png"."""
    return SPLIT_FILE.validate_json((BENCHMARK / f"photos_{name}.json").read_bytes())


def build(cfg: BuildConfig) -> None:
    """Recreate the crops and the splits from COCO and OpenImages."""
    labels = load_labels()
    photos = deduplicate(crop_all(labels, cfg.workers))
    splits = split(photos, cfg.seed, cfg.test_n, cfg.dev_n)
    BENCHMARK.mkdir(exist_ok=True)
    for name, part in splits.items():
        rows = {p.key: p.time for p in sorted(part, key=lambda p: p.key)}
        (BENCHMARK / f"photos_{name}.json").write_bytes(SPLIT_FILE.dump_json(rows, indent=1))
    print(json.dumps({"labelled": len(labels), "unique": len(photos), **{k: len(v) for k, v in splits.items()}}))


def push(cfg: HubConfig) -> None:
    """Upload the crops so other machines skip the build."""
    for name in SPLITS:
        labels = load_split(name)
        keys = sorted(labels)
        rows = {
            "key": keys,
            "image": [str(CROPS / f"{k}.png") for k in keys],
            "hours": [labels[k].hours for k in keys],
            "minutes": [labels[k].minutes for k in keys],
        }
        dataset = Dataset.from_dict(rows).cast_column("image", ImageColumn())
        dataset.push_to_hub(cfg.dataset, split=name, private=cfg.private)
        print(f"pushed {len(keys)} crops as {name}")


def pull(cfg: HubConfig) -> None:
    """Download the crops that `build` produced."""
    CROPS.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        dataset = load_dataset(cfg.dataset, split=name)
        for row in tqdm(dataset, total=len(dataset), desc=name):
            row["image"].save(CROPS / f"{row['key']}.png")
        print(f"pulled {len(dataset)} crops for {name}")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"build": build, "push": push, "pull": pull})
