"""Current eeglab_tests coverage for importing event tables."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc.pop_importevent import pop_importevent
from tests.eeglab_tests import eeglab_test


UPSTREAM = "unittesting_popfunc/pop_importevent/popfunc_pop_importevent_wrapperTest.m"


def _continuous_eeg() -> dict:
    eeg = eeg_from_data(np.zeros((1, 25_000), dtype=float), srate=100)
    old_events = [
        {"type": f"old-{index + 1}", "latency": 101.0 + index * 50.25, "urevent": index} for index in range(200)
    ]
    eeg["event"] = old_events
    eeg["urevent"] = [{key: value for key, value in event.items() if key != "urevent"} for event in old_events]
    return eeg


def _write_event_table(path: Path, delimiter: str) -> Path:
    rows = [delimiter.join(("Latency", "Type", "Position"))]
    rows.extend(
        delimiter.join((str(index), "target" if index % 2 else "response", str(index % 2 + 1)))
        for index in range(1, 101)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


_CASES = [
    (1, (), 300, 100, 101.0, 10_001.0),
    (2, ("append", "no", "optimalign", "on"), 100, 100, 101.0, 10_001.0),
    (3, ("append", "no", "optimalign", "off"), 100, 100, 101.0, 10_001.0),
    (4, ("append", "no", "align", 0, "optimalign", "on"), 100, 100, 101.0, 10_062.875),
    (5, ("append", "no", "align", 0, "optimalign", "off"), 100, 100, 102.0, 10_002.0),
    (6, ("append", "no", "align", 199, "optimalign", "on"), 100, 100, 10_101.75, 20_001.75),
    (7, ("append", "no", "align", 199, "optimalign", "off"), 100, 100, 10_101.75, 20_001.75),
    (8, ("append", "no", "align", -99, "optimalign", "on"), 1, 1, 101.0, 101.0),
    (9, ("append", "no", "align", -99, "optimalign", "off"), 2, 2, 2.0, 102.0),
    (10, ("append", "yes", "optimalign", "on"), 300, 100, 101.0, 10_001.0),
    (11, ("append", "yes", "optimalign", "off"), 300, 100, 101.0, 10_001.0),
    (12, ("append", "yes", "align", 0, "optimalign", "on"), 300, 100, 101.0, 10_001.0),
    (13, ("append", "yes", "align", 0, "optimalign", "off"), 300, 100, 101.0, 10_001.0),
    (14, ("append", "yes", "align", 199, "optimalign", "on"), 300, 100, 10_100.75, 20_000.75),
    (15, ("append", "yes", "align", 199, "optimalign", "off"), 300, 100, 10_100.75, 20_000.75),
    (16, ("append", "yes", "align", -99, "optimalign", "on"), 251, 51, 26.25, 5_026.25),
    (17, ("append", "yes", "align", -99, "optimalign", "off"), 251, 51, 26.25, 5_026.25),
    (18, ("append", "no", "indices", list(range(1, 101)), "optimalign", "on"), 200, 100, 101.0, 10_001.0),
    (19, ("append", "no", "indices", [], "optimalign", "on"), 100, 100, 101.0, 10_001.0),
    (20, ("append", "no", "timeunit", 1e-3, "optimalign", "on"), 100, 100, 1.1, 11.0),
    (21, ("delim", ","), 300, 100, 101.0, 10_001.0),
]


@eeglab_test(UPSTREAM, "test_test_pop_importevent")
@pytest.mark.parametrize(
    ("case", "extra_options", "event_count", "imported_count", "first_latency", "last_latency"),
    _CASES,
    ids=[f"case-{case}" for case, *_rest in _CASES],
)
def test_current_pop_importevent_option_cases(
    tmp_path: Path,
    case: int,
    extra_options: tuple,
    event_count: int,
    imported_count: int,
    first_latency: float,
    last_latency: float,
) -> None:
    delimiter = "," if case == 21 else "\t"
    event_file = _write_event_table(tmp_path / f"events-{case}.txt", delimiter)
    options = [
        "event",
        event_file,
        "fields",
        ["latency", "type", "position"],
        "skipline",
        1,
        "timeunit",
        1,
        "align",
        math.nan,
    ]
    options.extend(extra_options)

    output = pop_importevent(_continuous_eeg(), *options)

    events = [dict(event) for event in output["event"]]
    imported = [event for event in events if event.get("type") in {"target", "response"}]
    assert len(events) == event_count
    assert len(imported) == imported_count
    assert imported[0]["latency"] == pytest.approx(first_latency, abs=1e-3)
    assert imported[-1]["latency"] == pytest.approx(last_latency, abs=1e-3)
    assert all(event["position"] in {1, 2} for event in imported)
    assert all(event["init_index"] in range(1, 101) for event in imported)
    assert all(output["urevent"][event["urevent"]]["type"] == event["type"] for event in events)
