from __future__ import annotations

from timewizard.photos import Photo, Source, load_split, split
from timewizard.reading import Time


def fake_photos(n_coco: int, n_openimages: int) -> list[Photo]:
    def make(source: Source, name: str, i: int) -> Photo:
        return Photo(source=source, file_name=name, bbox=(0, 0, 10, 10), time=Time(hours=1 + i % 12, minutes=i % 60))

    coco = [make("coco", f"{i:012d}.jpg", i) for i in range(n_coco)]
    openimages = [make("openimages", f"{i:016x}.jpg", i) for i in range(n_openimages)]
    return coco + openimages


def test_split_is_seeded_disjoint_and_stratified() -> None:
    photos = fake_photos(600, 400)
    first = split(photos, seed=0, test_n=200, dev_n=100)
    assert first == split(photos, seed=0, test_n=200, dev_n=100)
    keys = [p.key for part in first.values() for p in part]
    assert len(keys) == len(set(keys)) == 1000
    assert (len(first["test"]), len(first["dev"])) == (200, 100)
    assert sum(p.source == "coco" for p in first["test"]) == 120
    assert split(photos, seed=1, test_n=200, dev_n=100)["test"] != first["test"]


def test_photo_urls_cover_every_bucket_of_its_source() -> None:
    photo = Photo(source="coco", file_name="x.jpg", bbox=(0, 0, 1, 1), time=Time(hours=1, minutes=0))
    assert photo.key == "coco_x"
    assert photo.urls() == [
        "http://images.cocodataset.org/train2017/x.jpg",
        "http://images.cocodataset.org/val2017/x.jpg",
    ]


def test_frozen_splits_are_disjoint_and_sized() -> None:
    splits = {name: load_split(name) for name in ("test", "dev", "train")}
    assert {k: len(v) for k, v in splits.items()} == {"test": 200, "dev": 200, "train": 2796}
    keys = [k for part in splits.values() for k in part]
    assert len(keys) == len(set(keys))
