"""Integrity and live proof for the native statement-coverage instrumentation."""

import hashlib
import json
import shutil
from unittest.mock import Mock
from pathlib import Path

import pytest

from tools.eeglab_statement_coverage import (
    REQUIRED_PLUGINS,
    _baseline_record,
    _unmeasurable_approval,
    run_native,
    source_inventory,
    verify_manifest,
)


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


@pytest.mark.parametrize("same_tree", [True, False])
def test_nested_eeglab_layout_accepts_only_the_verified_runtime_tree(tmp_path, same_tree):
    source = _source_tree(tmp_path / "reference/eeglab")
    _write(source, "functions/epoch.m")
    manifest = {
        "suite_root": str(source.parent),
        "eeglab_root": str(source),
        "sources": source_inventory(source),
        "test_files": [],
    }
    manifest_path = _write(tmp_path, "manifest.json", json.dumps(manifest))
    test_root = tmp_path / "tests"
    nested_runtime = Path(shutil.copytree(source, test_root / "eeglab"))
    runtime = nested_runtime if same_tree else Path(shutil.copytree(source, tmp_path / "other-runtime"))
    # A matching real copy reaches the next input check without starting MATLAB.
    # Even a byte-identical but different tree must fail the layout check.
    message = "Select at least one" if same_tree else "not a different EEGLAB tree"
    with pytest.raises(ValueError, match=message):
        run_native(manifest_path, test_root, [], tmp_path / "output", "must-not-start", 1, [], runtime)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("aliased", ["test_root", "runtime_root", "output"])
def test_direct_runner_rejects_symlink_aliases_into_reference(tmp_path, aliased):
    source = _source_tree(tmp_path / "reference/eeglab")
    manifest = {"suite_root": str(source.parent), "eeglab_root": str(source), "sources": source_inventory(source)}
    manifest_path = _write(tmp_path, "manifest.json", json.dumps(manifest))
    alias = tmp_path / "reference_alias"
    alias.symlink_to(source.parent, target_is_directory=True)
    paths = {"test_root": tmp_path / "tests", "runtime_root": tmp_path / "runtime", "output": tmp_path / "output"}
    paths[aliased] = alias
    with pytest.raises(ValueError, match="pinned reference checkout"):
        run_native(
            manifest_path, paths["test_root"], [], paths["output"], "must-not-start", 1, [], paths["runtime_root"]
        )
    assert not (tmp_path / "output").exists()


def _runner_layout(tmp_path):
    source = _source_tree(tmp_path / "reference/eeglab")
    runtime = Path(shutil.copytree(source, tmp_path / "tests/eeglab"))
    manifest = {
        "suite_root": str(source.parent),
        "eeglab_root": str(source),
        "sources": source_inventory(source),
        "test_files": [],
        "test_sources": [],
        "metric": "statement",
    }
    manifest_path = _write(tmp_path, "manifest.json", json.dumps(manifest))
    return manifest_path, runtime.parent, runtime


@pytest.mark.parametrize("invalid", ["missing", "not_matlab", "reference", "duplicate"])
def test_additional_native_tests_reject_invalid_inputs_before_launch(tmp_path, invalid):
    manifest, test_root, runtime = _runner_layout(tmp_path)
    additional = [_write(tmp_path, "added/test_expansion.m")]
    messages = {"missing": "existing .m", "not_matlab": "existing .m", "reference": "separate", "duplicate": "distinct"}
    if invalid == "missing":
        additional = [tmp_path / "absent.m"]
    elif invalid == "not_matlab":
        additional = [_write(tmp_path, "added/test.txt")]
    elif invalid == "reference":
        additional = [_write(tmp_path, "reference/test_expansion.m")]
    else:
        additional.append(_write(tmp_path, "other/test_expansion.m"))
    with pytest.raises(ValueError, match=messages[invalid]):
        run_native(manifest, test_root, [], tmp_path / "output", "must-not-start", 1, [], runtime, additional)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("changed", ["source", "snapshot"])
def test_added_tests_are_frozen_separately_without_changing_denominator(tmp_path, monkeypatch, changed):
    manifest, test_root, runtime = _runner_layout(tmp_path)
    original_manifest = json.loads(manifest.read_text())
    content = "function tests = test_expansion\ntests = functiontests(localfunctions);\nend\n"
    addition = _write(tmp_path, "added/test_expansion.m", content)
    output = tmp_path / "output"

    def launch(*args, **kwargs):
        config = json.loads((output / "run.json").read_text())
        assert config["sources"] == original_manifest["sources"]
        assert config["test_files"] == config["test_sources"] == config["selected"] == []
        assert config["additional_test_sources"] == [
            {
                "source": str(addition),
                "snapshot": "additional_tests/test_expansion.m",
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        ]
        snapshot = output / config["additional_test_sources"][0]["snapshot"]
        assert snapshot.read_text() == content
        (addition if changed == "source" else snapshot).write_text("% changed after snapshot\n")
        return Mock(wait=Mock(return_value=0))

    monkeypatch.setattr("tools.eeglab_statement_coverage.subprocess.Popen", launch)
    if changed == "snapshot":
        with pytest.raises(ValueError, match="snapshot changed during execution"):
            run_native(manifest, test_root, [], output, "matlab", 1, [], runtime, [addition])
    else:
        assert run_native(manifest, test_root, [], output, "matlab", 1, [], runtime, [addition]) == 0
        assert (output / "additional_tests/test_expansion.m").read_text() == content
    assert json.loads(manifest.read_text()) == original_manifest


@pytest.mark.parametrize("changed", ["sources", "test_sources", "runtime", "incomplete", None])
def test_coverage_union_requires_matching_completed_baseline(tmp_path, changed):
    manifest_path, _, runtime = _runner_layout(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    config = {**manifest, "eeglab_root": str(runtime)}
    report = {
        "execution_complete": True,
        "matlab_version": "test release",
        "covered_statements": 4,
        "tests": [{"failed": False, "incomplete": False}],
        "baseline": {"native_failures": True},
    }
    if changed in {"sources", "test_sources"}:
        config[changed] = [{"path": "changed.m"}]
    elif changed == "runtime":
        config["eeglab_root"] = str(tmp_path / "other-runtime")
    elif changed == "incomplete":
        report["execution_complete"] = False
    baseline = tmp_path / "baseline"
    _write(baseline, "run.json", json.dumps(config))
    _write(baseline, "report.json", json.dumps(report))
    _write(baseline, "coverage.mat", "retained native artifact")
    if changed is not None:
        with pytest.raises(ValueError):
            _baseline_record(baseline, manifest, runtime)
    else:
        record = _baseline_record(baseline, manifest, runtime)
        assert record["native_failures"]  # A green later batch cannot erase an earlier failure.
        assert record["covered_statements"] == 4
        assert set(record["artifacts"]) == {"run.json", "report.json", "coverage.mat"}


@pytest.mark.parametrize("changed", ["hash", "missing", "dependency", "duplicate", None])
def test_unmeasurable_approval_is_bound_to_distinct_frozen_source_hashes(tmp_path, changed):
    root = _source_tree(tmp_path / "runtime")
    _write(root, "functions/invalid.m")
    _write(root, "plugins/clean_rawdata/manopt/external.m")
    manifest = {"sources": source_inventory(root)}
    source = next(item for item in manifest["sources"] if item["path"] == "functions/invalid.m")
    approved = {"path": source["path"], "sha256": source["sha256"]}
    if changed == "hash":
        approved["sha256"] = "0" * 64
    elif changed == "missing":
        approved["path"] = "functions/absent.m"
    elif changed == "dependency":
        approved = next(item for item in manifest["sources"] if item["category"] == "dependency")
    payload = {"sources": [approved, approved] if changed == "duplicate" else [approved]}
    path = _write(tmp_path, "approval.json", json.dumps(payload))
    if changed:
        with pytest.raises(ValueError):
            _unmeasurable_approval(path, manifest)
    else:
        assert _unmeasurable_approval(path, manifest)["sources"] == [approved]


@pytest.mark.matlab
@pytest.mark.parametrize("invalid_source,approved_invalid", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("use_baseline", [False, True])
def test_native_statement_metric_keeps_failures_and_never_executed_files(
    eeglab_matlab_engine, tmp_path, invalid_source, approved_invalid, use_baseline
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
    output = tmp_path / "output"
    _write(
        output,
        "additional_tests/coverage_second_test.m",
        "function tests = coverage_second_test\ntests = functiontests(localfunctions);\nend\n"
        "function test_other_branch(testCase)\nverifyEqual(testCase, coverage_called(-2), -5);\nend\n",
    )
    config = {
        "eeglab_root": str(source),
        "sources": source_inventory(source),
        "test_root": str(test_root),
        "output": str(output),
        "test_files": ["coverage_probe_test.m", "not_selected.m"],
        "selected": ["coverage_probe_test.m"],
        "additional_test_sources": [{"snapshot": "additional_tests/coverage_second_test.m"}],
        "baseline": {},
        "unmeasurable_approval": {
            "matlab_version": eeglab_matlab_engine.version(),
            "sources": [{"path": "coverage_invalid.m"}],
        }
        if approved_invalid
        else {},
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
        failure = (
            "Unmeasurable frozen source" if invalid_source and not approved_invalid else "Native failures/incomplete"
        )
        if use_baseline:
            # First measure only the positive branch. The next invocation has
            # additions only, repeats three hits (including the if predicate),
            # and must contribute just one.
            config["additional_test_sources"] = []
            config_path.write_text(json.dumps(config))
            with pytest.raises(Exception, match=failure):
                engine.eeglab_statement_coverage(str(config_path), nargout=0)
            prior = json.loads((output / "report.json").read_text())
            assert prior["covered_statements"] == 4
            baseline_directory = output
            output = tmp_path / "added-output"
            shutil.copytree(baseline_directory / "additional_tests", output / "additional_tests")
            config.update(
                output=str(output),
                selected=[],
                additional_test_sources=[{"snapshot": "additional_tests/coverage_second_test.m"}],
                baseline={
                    "directory": str(baseline_directory),
                    "matlab_version": prior["matlab_version"],
                    "covered_statements": 4,
                    "native_failures": True,
                },
            )
            config_path.write_text(json.dumps(config))
        with pytest.raises(Exception, match=failure):
            engine.eeglab_statement_coverage(str(config_path), nargout=0)
    finally:
        engine.path(old_path, nargout=0)
    report = json.loads((output / "report.json").read_text())
    assert report["denominator_complete"] is not invalid_source
    assert report["approved_denominator_complete"] == (not invalid_source or approved_invalid)
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
        assert report["approved_unmeasurable_files"] == (["coverage_invalid.m"] if approved_invalid else [])
        assert report["unapproved_unmeasurable_files"] == ([] if approved_invalid else ["coverage_invalid.m"])
    assert sum(item["failed"] for item in report["tests"]) == (0 if use_baseline else 1)
    assert sum(item["passed"] for item in report["tests"]) == (1 if use_baseline else 2)
    assert len(report["discovered_test_cases"]) == (1 if use_baseline else 3)
    if use_baseline:
        assert report["batch_covered_statements"] == 4
        assert report["newly_covered_statements"] == 1
        assert report["baseline"]["native_failures"]
        assert report["unselected_test_files"] == ["coverage_probe_test.m", "not_selected.m"]
        assert report["completed_test_files"] == ["additional_tests/coverage_second_test.m"]
    else:
        assert report["unselected_test_files"] == ["not_selected.m"]
        assert report["completed_test_files"] == ["coverage_probe_test.m", "additional_tests/coverage_second_test.m"]
    assert (output / "coverage.mat").is_file()
    assert (output / "results.xml").is_file()
