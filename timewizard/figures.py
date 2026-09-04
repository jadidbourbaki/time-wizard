"""Figures for the model card, drawn from the score files in `runs/bench`.

`accuracy.png` compares every model on the test split. `cost.png` plots
accuracy against the price of reading one clock.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import tyro
from pydantic import BaseModel

from timewizard.bench import Report
from timewizard.photos import REPO

matplotlib.use("Agg")

NAMES = {
    "jadidbourbaki/time-wizard": "Time Wizard",
    "bedrock-mantle:openai.gpt-5.6-sol": "GPT-5.6 Sol",
    "anthropic:claude-fable-5-1": "Claude Fable 5.1",
    "anthropic:claude-opus-5": "Claude Opus 5",
}
PRICES = {
    "bedrock-mantle:openai.gpt-5.6-sol": (4.0, 20.0),
    "anthropic:claude-fable-5-1": (10.0, 50.0),
    "anthropic:claude-opus-5": (5.0, 25.0),
}
"""Dollars per million input and output tokens, list prices on 2026-09-04."""
H100_DOLLARS_PER_HOUR = 3.85
"""Nebius price for the H100 that scores time-wizard."""


class FigureConfig(BaseModel):
    runs: Path = REPO / "runs" / "bench"
    out: Path = REPO / "model"
    gpu_seconds: float | None = None
    """Wall clock for time-wizard on the 200 test clocks on one H100. None skips the cost figure."""


def load_reports(runs: Path) -> list[Report]:
    reports = [Report.model_validate_json(p.read_text()) for p in sorted(runs.glob("test_*.score.json"))]
    return sorted(reports, key=lambda r: list(NAMES).index(r.model))


def standard_error(share: float, n: int) -> float:
    return math.sqrt(share * (1 - share) / n)


def cost_per_clock(report: Report, gpu_seconds: float | None) -> float | None:
    if report.model in PRICES:
        per_input, per_output = PRICES[report.model]
        return (report.input_tokens * per_input + report.output_tokens * per_output) / 1e6 / report.score.n
    if gpu_seconds is not None:
        return gpu_seconds / 3600 * H100_DOLLARS_PER_HOUR / report.score.n
    return None


def accuracy_figure(reports: list[Report], out: Path) -> None:
    names = [NAMES[r.model] for r in reports]
    n = reports[0].score.n
    minute = [100 * r.score.within_tolerance for r in reports]
    hour = [100 * r.score.hour_correct for r in reports]
    minute_err = [100 * standard_error(r.score.within_tolerance, n) for r in reports]
    hour_err = [100 * standard_error(r.score.hour_correct, n) for r in reports]
    x = range(len(reports))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax.bar([i - width / 2 for i in x], minute, width, yerr=minute_err, capsize=4, label="Within 1 min", color="#1f4e79")
    ax.bar([i + width / 2 for i in x], hour, width, yerr=hour_err, capsize=4, label="Hour correct", color="#9dbcd4")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (m, h) in enumerate(zip(minute, hour, strict=True)):
        ax.text(i - width / 2, m + 4, f"{m:.1f}", ha="center", fontsize=9)
        ax.text(i + width / 2, h + 4, f"{h:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png")


def cost_figure(reports: list[Report], out: Path, gpu_seconds: float | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    for r in reports:
        cost = cost_per_clock(r, gpu_seconds)
        if cost is None:
            continue
        cents = 100 * cost
        y = 100 * r.score.within_tolerance
        ax.scatter(cents, y, s=80, color="#1f4e79", zorder=3)
        ax.annotate(NAMES[r.model], (cents, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 100)
    ax.set_ylim(0, 100)
    ticks = [0.01, 0.1, 1, 10, 100]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xlabel("Cents per clock")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, which="major", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "cost.png")


def main(cfg: FigureConfig) -> None:
    reports = load_reports(cfg.runs)
    cfg.out.mkdir(parents=True, exist_ok=True)
    accuracy_figure(reports, cfg.out)
    if cfg.gpu_seconds is not None:
        cost_figure(reports, cfg.out, cfg.gpu_seconds)
    print(f"wrote figures for {len(reports)} models to {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(FigureConfig))
