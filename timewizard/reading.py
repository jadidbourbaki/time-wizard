"""The task: a clock image in, hours and minutes out, scored against a label."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

SYSTEM = "Reply with ONLY the requested JSON, no preface and no code block."
PROMPT = 'What time does this analog clock show? Reply as JSON: {"hours": H, "minutes": M} with H from 1 to 12.'
MINUTE_TOLERANCE = 1

_JSON = re.compile(r"\{.*\}", re.S)


class Time(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hours: int = Field(ge=1, le=12)
    minutes: int = Field(ge=0, le=59)

    def minute_of_dial(self) -> int:
        return (self.hours % 12) * 60 + self.minutes


class Score(BaseModel):
    n: int
    within_tolerance: float
    """Fraction within MINUTE_TOLERANCE minutes. The headline number."""
    exact: float
    hour_correct: float
    mean_error_minutes: float | None
    unparsed: int


def parse_time(reply: str) -> Time | None:
    """The first JSON object in a model reply, or None when there is none or it
    does not describe a valid time."""
    match = _JSON.search(reply)
    if match is None:
        return None
    try:
        return Time.model_validate_json(match.group(0))
    except ValueError:
        return None


def circular_error(pred: Time, truth: Time) -> int:
    """Minutes between two times on a 12 hour dial, at most 360."""
    diff = abs(pred.minute_of_dial() - truth.minute_of_dial())
    return min(diff, 720 - diff)


def score(preds: list[Time | None], truths: list[Time]) -> Score:
    pairs = [(p, t) for p, t in zip(preds, truths, strict=True) if p is not None]
    errors = [circular_error(p, t) for p, t in pairs]
    n = len(truths)
    return Score(
        n=n,
        within_tolerance=sum(e <= MINUTE_TOLERANCE for e in errors) / n,
        exact=sum(e == 0 for e in errors) / n,
        hour_correct=sum(p.hours == t.hours for p, t in pairs) / n,
        mean_error_minutes=sum(errors) / len(errors) if errors else None,
        unparsed=n - len(pairs),
    )
