import json

import numpy as np
import pytest

from eegprep import OracleBackend as TopLevelOracleBackend
from eegprep.parity import (
    ArtifactOracle,
    ComparisonResult,
    OracleBackend,
    compare_eeg_struct,
    compare_numeric,
    compare_visual_output,
    compare_workflow_trace,
    detect_oracle_backends,
    load_deviations,
    load_manifest,
    render_markdown_report,
    resolve_oracle_backend,
    write_json_report,
)


def test_default_manifest_loads():
    manifest = load_manifest()
    assert manifest.version >= 1
    assert len(manifest.cases) >= 1
    assert "tight_numeric" in manifest.tolerances
    assert manifest.default_backend == "artifact_oracle"


def test_default_deviations_load():
    deviations = load_deviations()
    assert len(deviations) >= 1
    assert all(dev.case_id for dev in deviations)


def test_manifest_summary_has_expected_keys():
    summary = load_manifest().summary()
    assert "case_count" in summary
    assert "tiers" in summary
    assert "surfaces" in summary


def test_compare_numeric_passes():
    result = compare_numeric(np.array([1.0, 2.0]), np.array([1.0, 2.0]), atol=1e-8)
    assert result.passed
    assert result.metrics["mismatch_count"] == 0


def test_compare_numeric_shape_mismatch_fails():
    result = compare_numeric(np.array([1.0]), np.array([[1.0]]))
    assert not result.passed
    assert "shape mismatch" in result.failures[0]


def test_compare_numeric_equal_nan_keeps_finite_metrics():
    result = compare_numeric(
        np.array([1.0, np.nan]),
        np.array([1.0, np.nan]),
        equal_nan=True,
    )
    assert result.passed
    assert result.metrics["mismatch_count"] == 0
    assert result.metrics["max_abs_diff"] == 0.0
    assert result.metrics["mean_abs_diff"] == 0.0
    assert result.metrics["rms_diff"] == 0.0


def test_compare_eeg_struct_ignores_default_path_fields():
    expected = {"data": np.array([[1.0]]), "filepath": "/tmp/a", "filename": "a.set"}
    actual = {"data": np.array([[1.0]]), "filepath": "/tmp/b", "filename": "b.set"}
    result = compare_eeg_struct(expected, actual)
    assert result.passed


def test_compare_workflow_trace_normalizes_whitespace():
    expected = ["pop_loadset( 'a.set' )", "pop_reref(EEG)"]
    actual = ["pop_loadset('a.set')", "  pop_reref(EEG)  "]
    result = compare_workflow_trace(expected, actual)
    assert result.passed


def test_compare_visual_output_supports_arrays():
    img = np.ones((4, 4, 3), dtype=np.float64)
    result = compare_visual_output(img, img.copy(), atol=0.0)
    assert result.passed


def test_oracle_backend_is_top_level_exported():
    assert TopLevelOracleBackend.ARTIFACT_ORACLE == OracleBackend.ARTIFACT_ORACLE


def test_detect_oracle_backends_reports_all_keys(tmp_path):
    availability = detect_oracle_backends(artifact_root=tmp_path)
    assert OracleBackend.LIVE_MATLAB_BATCH in availability
    assert OracleBackend.LIVE_MATLAB_ENGINE in availability
    assert OracleBackend.ARTIFACT_ORACLE in availability
    assert availability[OracleBackend.ARTIFACT_ORACLE].available


def test_detect_oracle_backends_reports_missing_artifact_root():
    availability = detect_oracle_backends()
    assert OracleBackend.ARTIFACT_ORACLE in availability
    assert not availability[OracleBackend.ARTIFACT_ORACLE].available
    assert "artifact_root" in availability[OracleBackend.ARTIFACT_ORACLE].detail


def test_resolve_artifact_oracle(tmp_path):
    oracle = resolve_oracle_backend(OracleBackend.ARTIFACT_ORACLE, artifact_root=tmp_path)
    assert isinstance(oracle, ArtifactOracle)


def test_resolve_artifact_oracle_requires_root():
    with pytest.raises(RuntimeError, match="ARTIFACT_ORACLE requires artifact_root"):
        resolve_oracle_backend(OracleBackend.ARTIFACT_ORACLE)


def test_artifact_oracle_rejects_path_traversal(tmp_path):
    oracle = ArtifactOracle(tmp_path)
    assert oracle.resolve("safe/report.json") == tmp_path / "safe" / "report.json"
    with pytest.raises(ValueError, match="escapes root"):
        oracle.resolve("../escape.json")
    with pytest.raises(ValueError, match="must be relative"):
        oracle.resolve(tmp_path / "absolute.json")


def test_compare_eeg_data_raises_on_shape_mismatch(monkeypatch):
    from eegprep.utils.stage_comparison import compare_eeg_data
    import eegprep.parity as parity_module

    failed_data = ComparisonResult(
        label="stage.data",
        passed=False,
        metrics={"expected_shape": (1,), "actual_shape": (1, 1)},
        failures=["shape mismatch: expected (1,), actual (1, 1)"],
    )
    failed_stage = ComparisonResult(
        label="stage_outputs",
        passed=False,
        children=[failed_data],
        failures=list(failed_data.failures),
    )
    monkeypatch.setattr(parity_module, "compare_stage_outputs", lambda *_args, **_kwargs: failed_stage)

    with pytest.raises(AssertionError, match="shape mismatch"):
        compare_eeg_data("py.set", "mat.set")


def test_resolve_prefers_batch_when_engine_unavailable(monkeypatch):
    from eegprep.parity import oracle as oracle_module

    monkeypatch.setattr(oracle_module.shutil, "which", lambda _: "/usr/bin/matlab")
    monkeypatch.setattr(
        oracle_module,
        "detect_oracle_backends",
        lambda artifact_root=None: {
            OracleBackend.LIVE_MATLAB_ENGINE: type("Avail", (), {"available": False, "detail": "missing"})(),
            OracleBackend.LIVE_MATLAB_BATCH: type("Avail", (), {"available": True, "detail": "/usr/bin/matlab"})(),
            OracleBackend.ARTIFACT_ORACLE: type("Avail", (), {"available": False, "detail": "artifact_root not provided"})(),
        },
    )
    oracle = resolve_oracle_backend(OracleBackend.LIVE_MATLAB_BATCH)
    assert oracle.backend_name == OracleBackend.LIVE_MATLAB_BATCH


def test_markdown_report_contains_backend_and_result():
    manifest = load_manifest()
    result = compare_numeric(np.array([1.0]), np.array([1.0]))
    report = render_markdown_report(
        title="Parity Smoke",
        manifest=manifest,
        result=result,
        backend="artifact_oracle",
    )
    assert "Parity Smoke" in report
    assert "artifact_oracle" in report
    assert '"passed": true' in report


def test_json_report_writes_file(tmp_path):
    result = compare_numeric(np.array([1.0]), np.array([1.0]))
    target = write_json_report(tmp_path / "report.json", result)
    payload = json.loads(target.read_text())
    assert payload["passed"]
