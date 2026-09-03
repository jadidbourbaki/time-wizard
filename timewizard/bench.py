"""Score a model on a frozen photo split.

Each clock is asked once. `--adapter` scores our fine-tuned model.
`--model` scores any model pydantic-ai can reach, which puts frontier
baselines on the same photographs.

The runner appends each reply to a JSON lines file as it arrives. A rerun
asks only about the missing clocks. An interrupted run therefore resumes
without repeating API spend. A clock the model cannot answer records a
bracketed reason and counts as unparsed.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import torch
import tyro
from peft import PeftModel
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from timewizard import BASE_MODEL
from timewizard.photos import CROPS, REPO, Split, load_split
from timewizard.reading import PROMPT, SYSTEM, Score, parse_time, score

API_MAX_TOKENS = 32000
"""Reasoning at max effort runs to thousands of tokens for one clock."""
API_TIMEOUT = 600.0
"""Seconds one request may take. tenacity retries it after that."""
Effort = Literal["low", "medium", "high", "xhigh", "max"]
Answer = Callable[[Image.Image], str]


class Reply(BaseModel):
    key: str
    reply: str


class Report(BaseModel):
    split: Split
    model: str
    effort: Effort | None
    score: Score
    input_tokens: int
    output_tokens: int


class BenchConfig(BaseModel):
    split: Split = "dev"
    """Grade test once per final model, at the end."""
    adapter: Path | None = None
    """LoRA adapter directory for our fine-tuned model."""
    model: str | None = None
    """pydantic-ai model id, e.g. anthropic:claude-fable-5-1 or bedrock-mantle:openai.gpt-5.6-sol."""
    effort: Effort = "max"
    """Reasoning effort for API models."""
    limit: int = Field(0, ge=0)
    """Score only the first N photos of the split. Zero means all."""
    image_size: int = Field(448, ge=64)
    parallel: int = Field(8, ge=1)
    """Concurrent requests for API models."""
    out: Path = REPO / "runs" / "bench"
    """Replies are appended here as JSON lines and reused on a rerun."""


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def local_answer(adapter: Path, base: str = BASE_MODEL) -> Answer:
    processor = AutoProcessor.from_pretrained(adapter, max_image_tokens=256)
    model = PeftModel.from_pretrained(
        AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16), str(adapter)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    def answer(image: Image.Image) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()

    return answer


def api_answer(model: str, effort: Effort, usage: dict[str, int]) -> Answer:
    """Token counts accumulate on `usage`."""
    settings: Any
    if model.startswith("anthropic:"):
        settings = AnthropicModelSettings(anthropic_effort=effort, max_tokens=API_MAX_TOKENS, timeout=API_TIMEOUT)
    elif model.startswith("bedrock-mantle:"):
        settings = OpenAIResponsesModelSettings(
            openai_reasoning_effort=effort, max_tokens=API_MAX_TOKENS, timeout=API_TIMEOUT
        )
    else:
        raise SystemExit(f"unsupported model prefix in {model!r}; use anthropic: or bedrock-mantle:")
    agent = Agent(model, instructions=SYSTEM, model_settings=settings)

    @retry(
        retry=retry_if_exception_type(ModelAPIError),
        wait=wait_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def ask(image: Image.Image) -> str:
        result = agent.run_sync([PROMPT, BinaryContent(data=png_bytes(image), media_type="image/png")])
        usage["input_tokens"] += result.usage.input_tokens
        usage["output_tokens"] += result.usage.output_tokens
        return str(result.output).strip()

    def answer(image: Image.Image) -> str:
        try:
            return ask(image)
        except (ModelAPIError, UnexpectedModelBehavior) as err:
            return f"[no answer: {err}]"

    return answer


def collect(keys: list[str], answer: Answer, path: Path, parallel: int, image_size: int) -> dict[str, str]:
    """Answer every key missing from `path`, appending each reply as it arrives."""
    done: dict[str, str] = {}
    if path.exists():
        done = {r.key: r.reply for r in map(Reply.model_validate_json, path.read_text().splitlines())}
    todo = [k for k in keys if k not in done]

    def run(key: str) -> Reply:
        with Image.open(CROPS / f"{key}.png") as image:
            square = image.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
        return Reply(key=key, reply=answer(square))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f, ThreadPoolExecutor(parallel) as pool:
        for reply in tqdm(pool.map(run, todo), total=len(todo), desc=path.stem):
            f.write(reply.model_dump_json() + "\n")
            f.flush()
            done[reply.key] = reply.reply
    return done


def main(cfg: BenchConfig) -> None:
    usage = {"input_tokens": 0, "output_tokens": 0}
    if cfg.adapter is not None:
        name, answer, parallel = str(cfg.adapter), local_answer(cfg.adapter), 1
    elif cfg.model is not None:
        name, answer, parallel = cfg.model, api_answer(cfg.model, cfg.effort, usage), cfg.parallel
    else:
        raise SystemExit("pass --adapter or --model")

    labels = load_split(cfg.split)
    keys = sorted(labels)[: cfg.limit or None]
    path = cfg.out / f"{cfg.split}_{name.replace('/', '_').replace(':', '_')}.jsonl"
    replies = collect(keys, answer, path, parallel, cfg.image_size)
    report = Report(
        split=cfg.split,
        model=name,
        effort=cfg.effort if cfg.model else None,
        score=score([parse_time(replies[k]) for k in keys], [labels[k] for k in keys]),
        **usage,
    )
    path.with_suffix(".score.json").write_text(report.model_dump_json(indent=1))
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))
