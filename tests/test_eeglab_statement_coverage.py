"""Integrity and live proof for the native statement-coverage instrumentation."""

import json
import shutil
from pathlib import Path

import pytest

from tools.eeglab_statement_coverage import REQUIRED_PLUGINS, run_native, source_inventory, verify_manifest


def _write(root, name, text="value = 1;\n"):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def _source_tree(root):
    for name in REQUIRED_PLUGINS:
        (root / "plugins" / name).mkdir(parents=True)
    return root


def test_fixed_scope_includes_uncalled_private_gui_and_importer_implementations(tmp_path):
    root = _source_tree(tmp_path)
    names = [
        "eeglab.m",
        "functions/guifunc/unexecuted.m",
        "functions/@eegobj/private/helper.m",
        "plugins/Biosig3.8.5/internal/unexecuted.m",
        "plugins/Fileio260210/private/helper.m",
        "plugins/ICLabel/viewprops/pop_prop_extended.m",
        "plugins/bids-matlab-tools8.0/pop_importbids.m",
    ]
    for name in names:
        _write(root, name)
    _write(root, "plugins/clean_rawdata/manopt/manifold.m")
    _write(root, "plugins/ICLabel/matconvnet/matlab/layer.m")
    _write(root, "plugins/Fieldtrip-lite250523/ft_defaults.m")
    _write(root, "plugins/corrmap2.1/corrmap.m")
    _write(root, "plugins/EEG-BIDS/JSONio/jsonread.mexmaca64")
    inventory = {item["path"]: item for item in source_inventory(root)}
    assert {name for name, item in inventory.items() if item["category"] == "statement_scope"} == set(names)
    assert inventory["plugins/clean_rawdata/manopt/manifold.m"]["category"] == "dependency"
    assert inventory["plugins/ICLabel/matconvnet/matlab/layer.m"]["category"] == "dependency"
    assert inventory["plugins/Fieldtrip-lite250523/ft_defaults.m"]["category"] == "dependency"
    assert inventory["plugins/corrmap2.1/corrmap.m"]["category"] == "outside_approved_plugin_scope"
    assert inventory["plugins/EEG-BIDS/JSONio/jsonread.mexmaca64"]["category"] == "native_or_opaque"


@pytest.mark.parametrize("change", ["edit", "add", "remove"])
def test_frozen_denominator_rejects_source_drift(tmp_path, change):
    root = _source_tree(tmp_path)
    source = _write(root, "functions/unexecuted.m")
    manifest = {"eeglab_root": str(root), "sources": source_inventory(root)}
    verify_manifest(manifest)
    if change == "edit":
        source.write_text("value = 2;\n")
    elif change == "add":
        _write(root, "functions/another.m")
    else:
        source.unlink()
    with pytest.raises(ValueError, match="inventory changed"):
        verify_manifest(manifest)


def test_scope_refuses_missing_approved_plugin(tmp_path):
    with pytest.raises(ValueError, match="Missing approved plugins"):
        source_inventory(tmp_path)


def test_runtime_copy_has_identical_normalized_denominator_and_detects_edits(tmp_path):
    source = _source_tree(tmp_path / "reference")
    _write(source, "functions/eeg_checkset.m")
    manifest = {"eeglab_root": str(source), "sources": source_inventory(source)}
    copied = Path(shutil.copytree(source, tmp_path / "runtime"))
    verify_manifest(manifest, copied)
    _write(copied, "functions/eeg_checkset.m", "value = 3;\n")
    with pytest.raises(ValueError, match="inventory changed"):
        verify_manifest(manifest, copied)


@pytest.mark.parametrize(
    "metadata", ["Eeglab_tests.prj", "resources/project/Root.type.Project.xml", ".SimulinkProject/project.xml"]
)
def test_runner_rejects_project_metadata_before_startup(tmp_path, metadata):
    source = _source_tree(tmp_path / "reference/eeglab")
    manifest = {"suite_root": str(source.parent), "eeglab_root": str(source), "sources": source_inventory(source)}
    manifest_path = _write(tmp_path, "manifest.json", json.dumps(manifest))
    test_root = tmp_path / "tests"
    _write(test_root, metadata)
    with pytest.raises(ValueError, match="projectless scratch"):
        run_native(manifest_path, test_root, [], tmp_path / "output", "must-not-start", 1, [], tmp_path / "runtime")
    assert not (tmp_path / "output").exists()


@pytest.mark.matlab
@pytest.mark.parametrize("invalid_source", [False, True])
def test_native_statement_metric_keeps_failures_and_never_executed_files(
    eeglab_matlab_engine, tmp_path, invalid_source
):
    source = _source_tree(tmp_path / "source")
    # Only startup I/O is inert in this toy source tree; coverage_called below
    # really executes in MATLAB and includes both taken and untaken statements.
    _write(source, "eeglab.m", "function eeglab(varargin)\nend\n")
    _write(
        source,
        "coverage_called.m",
        "function value = coverage_called(value)\n"
        "value = value + 1; value = value * 2;\nif value > 0\nvalue = value + 3;\n"
        "else\nvalue = value - 3;\nend\nend\n",
    )
    _write(source, "coverage_uncalled.m", "function value = coverage_uncalled\nvalue = 1; value = value + 1;\nend\n")
    if invalid_source:
        _write(source, "coverage_invalid.m", "function value = coverage_invalid(\nvalue = ;\nend\n")
    test_root = tmp_path / "tests"
    _write(
        test_root,
        "coverage_probe_test.m",
        "function tests = coverage_probe_test\ntests = functiontests(localfunctions);\nend\n"
        "function test_pass(testCase)\nverifyEqual(testCase, coverage_called(1), 7);\nend\n"
        "function test_failure(testCase)\nverifyEqual(testCase, coverage_called(1), -1);\nend\n",
    )
    _write(
        test_root,
        "coverage_second_test.m",
        "function tests = coverage_second_test\ntests = functiontests(localfunctions);\nend\n"
        "function test_other_branch(testCase)\nverifyEqual(testCase, coverage_called(-2), -5);\nend\n",
    )
    output = tmp_path / "output"
    output.mkdir()
    config = {
        "eeglab_root": str(source),
        "sources": source_inventory(source),
        "test_root": str(test_root),
        "output": str(output),
        "test_files": ["coverage_probe_test.m", "coverage_second_test.m", "not_selected.m"],
        "selected": ["coverage_probe_test.m", "coverage_second_test.m"],
        "support_paths": [],
        "worker_coverage": "not verified",
        "home_options": "",
    }
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config))
    engine = eeglab_matlab_engine
    old_path = engine.path()
    try:
        engine.addpath(str(Path(__file__).resolve().parents[1] / "tools"), nargout=0)
        failure = "Unmeasurable frozen source" if invalid_source else "Native failures/incomplete"
        with pytest.raises(Exception, match=failure):
            engine.eeglab_statement_coverage(str(config_path), nargout=0)
    finally:
        engine.path(old_path, nargout=0)
    report = json.loads((output / "report.json").read_text())
    assert report["denominator_complete"] is not invalid_source
    assert report["execution_complete"]
    assert report["covered_statements"] == 5  # Union, not sum, of overlapping executions in two suites.
    assert report["total_statements"] == 7  # Two statements share one source line.
    files = {item["path"]: item for item in report["files"]}
    assert files["coverage_called.m"]["total_statements"] == 5
    assert files["coverage_uncalled.m"]["covered_statements"] == 0
    assert files["coverage_uncalled.m"]["total_statements"] == 2
    if invalid_source:
        assert report["unmeasurable_files"] == ["coverage_invalid.m"]
        assert files["coverage_invalid.m"]["invalid"]
        assert report["measurable_file_count"] == report["scope_file_count"] - 1
    assert sum(item["failed"] for item in report["tests"]) == 1
    assert sum(item["passed"] for item in report["tests"]) == 2
    assert len(report["discovered_test_cases"]) == 3
    assert report["unselected_test_files"] == ["not_selected.m"]
    assert (output / "coverage.mat").is_file()
    assert (output / "results.xml").is_file()
