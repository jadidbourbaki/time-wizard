from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import timewizard.bench as bench


def test_collect_appends_replies_and_skips_finished_clocks(tmp_path: Path) -> None:
    bench.CROPS = tmp_path
    for key in ("a", "b"):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{key}.png")
    path = tmp_path / "replies.jsonl"
    path.write_text(json.dumps({"key": "a", "reply": '{"hours": 3, "minutes": 30}'}) + "\n")
    asked: list[str] = []

    def answer(image: Image.Image) -> bench.Answer:
        asked.append("b")
        return bench.Answer(reply='{"hours": 9, "minutes": 15}', input_tokens=300, output_tokens=12)

    replies = bench.collect(["a", "b"], answer, path, parallel=1, image_size=8)
    assert asked == ["b"]
    assert set(replies) == {"a", "b"}
    assert replies["a"].output_tokens == 0
    assert replies["b"].output_tokens == 12
    assert len(path.read_text().splitlines()) == 2
