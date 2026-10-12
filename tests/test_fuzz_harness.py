"""Harness tests for differential fuzzing. MATLAB is not required."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import strategies as st

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.fuzz.harness import (
    EEGSpec,
    FuzzCase,
    Outcome,
    SideResult,
    Tolerance,
    compare_data,
    data_sha256,
    default_sanity,
    draw_cases,
    freeze_params,
    judge,
    materialize_eeg,
    nightly_seed,
    run_repro,
    run_session,
    side_from_error,
)
from tests.fuzz.issues import (
    FUNCTION_ROWS,
    DASHBOARD_MARKER,
    GithubIssueTracker,
    MemoryIssueTracker,
    OpenIssue,
    SIGNATURE_MARKER,
    publish_findings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec(n_channels: int = 1) -> EEGSpec:
    return EEGSpec(tuple(range(n_channels)), 0, 8, 1, -1, 0, 0)


def _reref_case(n_channels: int = 1, *, sanity: bool = False) -> FuzzCase:
    return FuzzCase(
        "pop_reref",
        _spec(n_channels),
        freeze_params({"ref": (), "keepref": "off", "exclude": ()}),
        sanity=sanity,
    )


def _ones(case: FuzzCase) -> dict:
    n_channels = len(case.spec.channels)
    return {"data": np.ones((n_channels, 8), dtype=float), "srate": 128.0}


class _Backend:
    def __init__(self, fn):
        self.fn = fn

    def run(self, case, eeg):
        try:
            data = np.asarray(self.fn(case, eeg), dtype=float)
            return SideResult(True, data, float(eeg["srate"]), None, None)
        except Exception as exc:
            return side_from_error(exc)


def _identity(case, eeg):
    del case
    return np.array(eeg["data"], dtype=float, copy=True)


def _diverge_over_two_channels(case, eeg):
    data = np.array(eeg["data"], dtype=float, copy=True)
    if len(case.spec.channels) > 2:
        data += 5
    return data


def _counts():
    return {name: {"examples": 4, "divergences": 0} for name in FUNCTION_ROWS}


def _finding(function: str, max_abs: float) -> Outcome:
    case = _reref_case()
    case = FuzzCase(function, case.spec, case.params)
    outcome = Outcome(case, True, True, "numeric", max_abs, max_abs, "detail")
    outcome.repro_command = f"uv run python -m tests.fuzz reproduce repros/{outcome.signature}.json"
    outcome.confirmed = True
    return outcome


def test_nightly_seed_is_utc_date():
    assert nightly_seed(date(2026, 10, 12)) == 20261012


def test_tolerance_file_is_marked_human_owned():
    text = (REPO_ROOT / "tests" / "fuzz" / "tolerances.toml").read_text()
    agents = (REPO_ROOT / "AGENTS.md").read_text()
    assert "Tolerances are changed only by reviewed human pull requests." in text
    assert "Agents must not loosen" in text
    assert "tests/fuzz/tolerances.toml" in agents
    assert "Agents must not loosen" in agents


def test_compare_respects_tolerance_shape_and_nan_mask():
    tolerance = Tolerance(1e-6, 1e-6)
    oracle = np.array([[1.0, 2.0]])
    matched = compare_data(oracle + 1e-9, oracle, 128.0, 128.0, tolerance)
    assert matched[0] is False

    diverged, error_class, max_abs, _max_rel, _detail = compare_data(oracle + 1e-2, oracle, 128.0, 128.0, tolerance)
    assert diverged is True
    assert error_class == "numeric"
    assert max_abs == pytest.approx(1e-2)

    diverged, error_class, *_rest = compare_data(np.zeros((1, 3)), np.zeros((1, 4)), 1.0, 1.0, tolerance)
    assert diverged is True
    assert error_class == "shape"

    nan_oracle = np.array([[1.0, np.nan]])
    nan_python = np.array([[1.0, 1.0]])
    diverged, error_class, *_rest = compare_data(nan_python, nan_oracle, 1.0, 1.0, tolerance)
    assert error_class == "nan-mask"

    both = judge(
        side_from_error(ValueError("no")),
        side_from_error(ValueError("no")),
        tolerance,
        _reref_case(),
        "abc",
    )
    assert both.diverged is False
    assert both.both_ok is False


def test_materialize_crops_decimates_and_injects_nans():
    case = FuzzCase(
        "pop_reref",
        EEGSpec(channels=(1, 3), start=4, length=5, decimate=2, nan_channel=0, nan_start=1, nan_length=2),
        freeze_params({"ref": (), "keepref": "off", "exclude": ()}),
    )
    eeg = materialize_eeg(case)
    source = np.asarray(pop_loadset(str(REPO_ROOT / "sample_data" / "eeglab_data.set"))["data"])
    expected = np.array(source[np.ix_([1, 3], np.arange(4, 14, 2))], dtype=float)
    expected[0, 1:3] = np.nan
    assert eeg["data"].shape == (2, 5)
    assert eeg["srate"] == 64.0
    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 5
    assert eeg["event"].size == 0
    np.testing.assert_allclose(eeg["data"], expected, equal_nan=True)
    assert len(data_sha256(eeg["data"])) == 64


def test_reref_sanity_signal_matches_across_float32_and_float64():
    case = next(item for item in default_sanity() if item.function == "pop_reref")
    data = np.asarray(materialize_eeg(case)["data"], dtype=np.float64)
    ref64 = data - data.mean(axis=0, keepdims=True)
    narrow = np.asarray(data, dtype=np.float32)
    ref32 = (narrow - narrow.mean(axis=0, keepdims=True)).astype(np.float64)
    assert np.max(np.abs(ref64 - ref32)) <= 1e-6


def test_same_seed_draws_the_same_cases():
    first = draw_cases("pop_reref", seed_value=20261012, examples=4)
    second = draw_cases("pop_reref", seed_value=20261012, examples=4)
    other = draw_cases("pop_reref", seed_value=20261013, examples=4)
    assert first == second
    assert first != other
    assert all(case.function == "pop_reref" for case in first)


def test_shrink_reduces_channel_count_to_the_failure_boundary(tmp_path):
    strategy = st.integers(1, 8).map(lambda n: _reref_case(n))
    result = run_session(
        seed_value=7,
        output_dir=tmp_path,
        examples=20,
        shrink_budget=40,
        max_shrinks=2,
        python_backend=_Backend(_identity),
        oracle_backend=_Backend(_diverge_over_two_channels),
        strategies={"pop_reref": strategy},
        sanity=[_reref_case(1, sanity=True)],
        materialize=_ones,
    )
    assert result.harness_ok
    assert result.functions["pop_reref"]["divergences"] >= 1
    assert result.divergences
    assert len(result.divergences[0].case.spec.channels) == 3
    repro = json.loads(next((tmp_path / "repros").glob("*.json")).read_text())
    assert repro["seed"] == 7
    assert repro["command"].startswith("uv run python -m tests.fuzz reproduce ")


def test_sanity_failure_does_not_file_divergence_issues(tmp_path):
    tracker = MemoryIssueTracker()

    def explode(case, eeg):
        del case, eeg
        raise RuntimeError("bridge down")

    result = run_session(
        seed_value=20261012,
        output_dir=tmp_path,
        examples=3,
        python_backend=_Backend(_identity),
        oracle_backend=_Backend(explode),
        strategies={"pop_reref": st.just(_reref_case(2))},
        sanity=[_reref_case(1, sanity=True)],
        materialize=_ones,
        tracker=tracker,
        run_url="https://github.com/sccn/eegprep/actions/runs/1",
    )
    assert result.harness_ok is False
    assert result.created == []
    assert result.divergences == []
    assert all(not issue.title.startswith("[fuzz]") for issue in tracker.issues)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["harness_ok"] is False
    assert metrics["seed"] == 20261012
    assert DASHBOARD_MARKER in tracker.issues[-1].body


def test_publish_dedupes_and_caps_new_issues():
    existing = _finding("pop_reref", 0.02)
    tracker = MemoryIssueTracker(
        [OpenIssue(4, "old", SIGNATURE_MARKER.format(signature=existing.signature) + "\n", ["fuzz"])]
    )
    findings = [
        existing,
        _finding("pop_eegfiltnew", 0.2),
        _finding("pop_resample", 2.0),
        _finding("clean_asr", 20.0),
        _finding("pop_reref", 3.0),
    ]
    # The fifth new signature shares a function with the existing one but not a magnitude.
    functions = _counts()
    for finding in findings:
        functions[finding.report]["divergences"] += 1
    result = publish_findings(
        seed=20261012,
        functions=functions,
        findings=findings,
        tracker=tracker,
        cap=3,
        run_url="https://github.com/sccn/eegprep/actions/runs/9",
        harness_ok=True,
    )
    assert result.commented == [4]
    assert len(result.created) == 3
    assert len(result.unfiled) == 1
    created = [issue for issue in tracker.issues if issue.number in result.created]
    assert all(issue.labels == ["fuzz", "fuzz-unconfirmed"] for issue in created)
    assert all("fuzz-signature:" in issue.body for issue in created)
    assert all("uv run python -m tests.fuzz reproduce" in issue.body for issue in created)
    assert "https://github.com/sccn/eegprep/actions/runs/9#artifacts" in created[0].body
    dashboard = next(issue for issue in tracker.issues if DASHBOARD_MARKER in issue.body)
    assert "pop_reref=" in dashboard.body
    again = publish_findings(
        seed=20261013,
        functions=functions,
        findings=[],
        tracker=tracker,
        cap=3,
        run_url="",
        harness_ok=True,
    )
    assert again.created == []
    body = next(issue for issue in tracker.issues if DASHBOARD_MARKER in issue.body).body
    assert "- 20261012:" in body
    assert "- 20261013:" in body


def test_github_tracker_sends_signature_and_labels():
    calls = []

    def runner(args):
        calls.append(list(args))
        if "list" in args:
            return "[]"
        if "create" in args:
            return "https://github.com/sccn/eegprep/issues/15\n"
        if "view" in args:
            return json.dumps({"id": "NODE"})
        return ""

    tracker = GithubIssueTracker("sccn/eegprep", "https://github.com/sccn/eegprep/actions/runs/3", runner)
    finding = _finding("pop_resample", 0.2)
    publish_findings(
        seed=20261012,
        functions=_counts(),
        findings=[finding],
        tracker=tracker,
        cap=3,
        run_url="https://github.com/sccn/eegprep/actions/runs/3",
        harness_ok=True,
    )
    bodies = []
    for args in calls:
        if "--body-file" in args:
            bodies.append(Path(args[args.index("--body-file") + 1]).read_text())
    assert any(f"fuzz-signature: {finding.signature}" in body for body in bodies)
    create = next(
        args for args in calls if "create" in args and "--title" in args and "[fuzz]" in args[args.index("--title") + 1]
    )
    assert create.count("--label") == 2
    assert "fuzz-unconfirmed" in create


def test_reproduce_reruns_and_rejects_a_changed_input_hash(tmp_path):
    python = _Backend(_identity)
    oracle = _Backend(_diverge_over_two_channels)
    result = run_session(
        seed_value=3,
        output_dir=tmp_path,
        examples=1,
        shrink_budget=2,
        python_backend=python,
        oracle_backend=oracle,
        strategies={"pop_reref": st.just(_reref_case(4))},
        sanity=[_reref_case(1, sanity=True)],
        materialize=_ones,
    )
    assert result.harness_ok
    repro = next((tmp_path / "repros").glob("*.json"))
    outcome = run_repro(repro, python_backend=python, oracle_backend=oracle, materialize=_ones)
    assert outcome.diverged is True
    payload = json.loads(repro.read_text())
    payload["input_sha256"] = "0" * 64
    repro.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="input hash"):
        run_repro(repro, python_backend=python, oracle_backend=oracle, materialize=_ones)


def test_cli_fake_match_on_sample_data_passes(tmp_path):
    from tests.fuzz.__main__ import main

    code = main(
        [
            "run",
            "--oracle",
            "fake",
            "--fake-mode",
            "match",
            "--examples",
            "1",
            "--shrink-budget",
            "1",
            "--output",
            str(tmp_path),
            "--seed",
            "20261012",
        ]
    )
    assert code == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["harness_ok"] is True
    assert metrics["seed"] == 20261012
    assert metrics["functions"]["pop_reref"]["divergences"] == 0
    assert metrics["functions"]["clean_asr"]["examples"] + metrics["functions"]["pop_clean_rawdata"]["examples"] == 1


def test_workflow_schedules_the_matlab_batch_job():
    text = (REPO_ROOT / ".github" / "workflows" / "fuzz-nightly.yml").read_text()
    assert 'cron: "23 10 * * *"' in text
    assert "timeout-minutes: 60" in text
    assert "issues: write" in text
    assert "MLM_LICENSE_TOKEN" in text
    assert "matlab-batch" in text
    assert "eegprep_batch_engine_serve" in text
