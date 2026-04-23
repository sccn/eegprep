import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eegprep.parity import (
    ArtifactOracle,
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


class TestParityConfig(unittest.TestCase):
    def test_default_manifest_loads(self):
        manifest = load_manifest()
        self.assertGreaterEqual(manifest.version, 1)
        self.assertGreaterEqual(len(manifest.cases), 1)
        self.assertIn("tight_numeric", manifest.tolerances)
        self.assertEqual(manifest.default_backend, "artifact_oracle")

    def test_default_deviations_load(self):
        deviations = load_deviations()
        self.assertGreaterEqual(len(deviations), 1)
        self.assertTrue(all(dev.case_id for dev in deviations))

    def test_manifest_summary_has_expected_keys(self):
        summary = load_manifest().summary()
        self.assertIn("case_count", summary)
        self.assertIn("tiers", summary)
        self.assertIn("surfaces", summary)


class TestParityCompare(unittest.TestCase):
    def test_compare_numeric_passes(self):
        result = compare_numeric(np.array([1.0, 2.0]), np.array([1.0, 2.0]), atol=1e-8)
        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["mismatch_count"], 0)

    def test_compare_numeric_shape_mismatch_fails(self):
        result = compare_numeric(np.array([1.0]), np.array([[1.0]]))
        self.assertFalse(result.passed)
        self.assertIn("shape mismatch", result.failures[0])

    def test_compare_eeg_struct_ignores_default_path_fields(self):
        expected = {"data": np.array([[1.0]]), "filepath": "/tmp/a", "filename": "a.set"}
        actual = {"data": np.array([[1.0]]), "filepath": "/tmp/b", "filename": "b.set"}
        result = compare_eeg_struct(expected, actual)
        self.assertTrue(result.passed)

    def test_compare_workflow_trace_normalizes_whitespace(self):
        expected = ["pop_loadset( 'a.set' )", "pop_reref(EEG)"]
        actual = ["pop_loadset('a.set')", "  pop_reref(EEG)  "]
        result = compare_workflow_trace(expected, actual)
        self.assertTrue(result.passed)

    def test_compare_visual_output_supports_arrays(self):
        img = np.ones((4, 4, 3), dtype=np.float64)
        result = compare_visual_output(img, img.copy(), atol=0.0)
        self.assertTrue(result.passed)


class TestParityOracle(unittest.TestCase):
    def test_detect_oracle_backends_reports_batch_key(self):
        availability = detect_oracle_backends()
        self.assertIn(OracleBackend.LIVE_MATLAB_BATCH, availability)
        self.assertIn(OracleBackend.LIVE_MATLAB_ENGINE, availability)

    def test_resolve_artifact_oracle(self):
        oracle = resolve_oracle_backend(OracleBackend.ARTIFACT_ORACLE, artifact_root=".")
        self.assertIsInstance(oracle, ArtifactOracle)

    @patch("eegprep.parity.oracle.detect_oracle_backends")
    def test_resolve_prefers_batch_when_engine_unavailable(self, mock_detect):
        mock_detect.return_value = {
            OracleBackend.LIVE_MATLAB_ENGINE: type("Avail", (), {"available": False, "detail": "missing"})(),
            OracleBackend.LIVE_MATLAB_BATCH: type("Avail", (), {"available": True, "detail": "/usr/bin/matlab"})(),
        }
        oracle = resolve_oracle_backend(OracleBackend.LIVE_MATLAB_BATCH)
        self.assertEqual(oracle.backend_name, OracleBackend.LIVE_MATLAB_BATCH)


class TestParityReporting(unittest.TestCase):
    def test_markdown_report_contains_backend_and_result(self):
        manifest = load_manifest()
        result = compare_numeric(np.array([1.0]), np.array([1.0]))
        report = render_markdown_report(
            title="Parity Smoke",
            manifest=manifest,
            result=result,
            backend="artifact_oracle",
        )
        self.assertIn("Parity Smoke", report)
        self.assertIn("artifact_oracle", report)
        self.assertIn('"passed": true', report)

    def test_json_report_writes_file(self):
        result = compare_numeric(np.array([1.0]), np.array([1.0]))
        with tempfile.TemporaryDirectory() as tmpdir:
            target = write_json_report(Path(tmpdir) / "report.json", result)
            payload = json.loads(target.read_text())
        self.assertTrue(payload["passed"])


if __name__ == "__main__":
    unittest.main()
