from __future__ import annotations

import pytest

from timewizard.reading import Time, circular_error, parse_time, score


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"hours": 3, "minutes": 7}', Time(hours=3, minutes=7)),
        ('Sure! ```json\n{"hours": 12, "minutes": 0, "seconds": 5}\n```', Time(hours=12, minutes=0)),
        ('{"hours": 0, "minutes": 7}', None),
        ("I cannot tell.", None),
    ],
)
def test_parse_time_extracts_the_first_valid_json_object(reply: str, expected: Time | None) -> None:
    assert parse_time(reply) == expected


def test_circular_error_wraps_at_twelve() -> None:
    assert circular_error(Time(hours=12, minutes=59), Time(hours=1, minutes=1)) == 2
    assert circular_error(Time(hours=6, minutes=0), Time(hours=12, minutes=0)) == 360


def test_score_counts_tolerance_exact_hours_and_unparsed() -> None:
    truths = [
        Time(hours=3, minutes=30),
        Time(hours=3, minutes=30),
        Time(hours=11, minutes=59),
        Time(hours=5, minutes=0),
    ]
    preds = [Time(hours=3, minutes=30), Time(hours=3, minutes=31), Time(hours=12, minutes=0), None]
    result = score(preds, truths)
    assert (result.n, result.within_tolerance, result.exact, result.hour_correct, result.unparsed) == (
        4,
        0.75,
        0.25,
        0.5,
        1,
    )
    assert result.mean_error_minutes == pytest.approx(2 / 3)
