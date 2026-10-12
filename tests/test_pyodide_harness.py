import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest

from eegprep.plugins.ICLabel.iclabel_net_onnx_web import run_iclabel_net_async
from tools import check_pyodide_base_resolution
from tools.pyodide import benchmark, compare_iclabel
from tools.pyodide.compare_benchmarks import compare_reports
from tools.pyodide.compare_iclabel import compare_reports as compare_iclabel_reports
from tools.pyodide.iclabel_parity import run_parity
from tools.pyodide.prepare_docopt_wheel import verify_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "a" * 64


def test_docopt_hash_check_rejects_modified_sdist(tmp_path):
    archive = tmp_path / "docopt-0.6.2.tar.gz"
    archive.write_bytes(b"locked sdist payload")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(archive, "0" * 64)


def test_benchmark_failure_records_the_full_traceback():
    calls = []

    def fails_after_warmup(_data):
        calls.append(1)
        if len(calls) > benchmark.WARMUP_RUNS:
            raise ValueError("measured run broke")
        return 1.0, 10, True

    summary = benchmark._ica_summary(
        platform="native", data=np.zeros((2, 2)), name="runica", operation=fails_after_warmup
    )

    error = summary["runs"][0]["error"]
    assert "Traceback (most recent call last)" in error
    assert "in fails_after_warmup" in error
    assert error.rstrip().endswith("ValueError: measured run broke")
    assert summary["median_seconds"] is None


def test_compare_reports_applies_convergence_and_blas_gates():
    def report(platform, runica_seconds, picard_seconds, picard_converged):
        matmul = []
        for case in ("activation", "weight_update"):
            for dtype, operation, blas_seconds in (
                ("float64", "dgemm", 0.5),
                ("float32", "sgemm", 0.9 if case == "activation" else 0.5),
            ):
                common = {
                    "case": case,
                    "dtype": dtype,
                    "platform": "pyodide",
                    "seed": 375,
                    "left_shape": [64, 64],
                    "right_shape": [64, 49] if case == "activation" else [64, 64],
                    "median_seconds": 1.0,
                }
                matmul.append({**common, "operation": "numpy_matmul"})
                matmul.append({**common, "operation": operation, "median_seconds": blas_seconds})
        return {
            "schema_version": 1,
            "platform": platform,
            "seed": 375,
            "shape": [64, 15_000],
            "runica_block": 49,
            "thread_count": 1,
            "ica": {
                "runica": {
                    "median_seconds": runica_seconds,
                    "median_iterations": 100,
                    "all_converged": True,
                },
                "picard": {
                    "median_seconds": picard_seconds,
                    "median_iterations": 50,
                    "all_converged": picard_converged,
                },
            },
            "matmul": matmul,
        }

    comparison = compare_reports(report("native", 10.0, 5.0, True), report("pyodide", 20.0, 4.0, True))

    assert comparison["ica"]["runica"]["pyodide_speed_ratio"] == 2.0
    assert comparison["ica"]["runica"]["native_median_iterations"] == 100
    assert comparison["ica"]["runica"]["native_all_converged"] is True
    assert comparison["matmul"][0]["native_blas_speedup_over_numpy"] == 2.0
    assert comparison["matmul"][0]["blas_speedup_over_numpy"] == 2.0
    assert comparison["decisions"]["picard_browser_default_retained"] is True
    assert comparison["decisions"]["blas_recommended"] is False


def _iclabel_report(platform, classifications, **overrides):
    values = np.asarray(classifications, dtype=np.float32)
    report = {
        "schema_version": compare_iclabel.SCHEMA_VERSION,
        "platform": platform,
        "dataset": "eeglab_data_with_ica_tmp.set",
        "model_sha256": MODEL_SHA256,
        "shape": list(values.shape),
        "classifications": values.tolist(),
    }
    return {**report, **overrides}


def _write_reports(tmp_path, native, pyodide):
    paths = {}
    for name, report in (("native", native), ("pyodide", pyodide)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(report))
    return ["--native", str(paths["native"]), "--pyodide", str(paths["pyodide"])]


def _two_component_values(offset=0.0):
    return np.asarray([[0.1] * 7, [0.9] * 7], dtype=np.float32) + np.float32(offset)


def test_compare_iclabel_main_accepts_difference_within_tolerance(tmp_path, capsys):
    arguments = _write_reports(
        tmp_path,
        _iclabel_report("native", _two_component_values()),
        _iclabel_report("pyodide", _two_component_values(1e-6)),
    )

    assert compare_iclabel.main(arguments) == 0

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["allclose"] is True
    assert comparison["max_absolute_difference"] <= compare_iclabel.ATOL
    assert comparison["model_sha256"] == MODEL_SHA256


def test_compare_iclabel_main_fails_difference_beyond_tolerance(tmp_path, capsys):
    report_path = tmp_path / "parity.md"
    arguments = _write_reports(
        tmp_path,
        _iclabel_report("native", _two_component_values()),
        _iclabel_report("pyodide", _two_component_values(1e-3)),
    )

    assert compare_iclabel.main([*arguments, "--report", str(report_path)]) == 1

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["allclose"] is False
    assert comparison["max_absolute_difference"] == pytest.approx(1e-3, rel=1e-2)
    assert "`allclose`: `False`" in report_path.read_text()


def test_compare_iclabel_reports_rejects_shape_mismatch():
    native = _iclabel_report("native", [[0.1] * 7])
    pyodide = _iclabel_report("pyodide", [[0.1] * 7, [0.9] * 7])

    with pytest.raises(ValueError, match="shapes differ"):
        compare_iclabel_reports(native, pyodide)


def test_compare_iclabel_reports_rejects_dataset_mismatch():
    native = _iclabel_report("native", _two_component_values())
    pyodide = _iclabel_report("pyodide", _two_component_values(), dataset="other.set")

    with pytest.raises(ValueError, match="different datasets"):
        compare_iclabel_reports(native, pyodide)


def test_compare_iclabel_reports_rejects_different_model_digests():
    native = _iclabel_report("native", _two_component_values())
    pyodide = _iclabel_report("pyodide", _two_component_values(), model_sha256="b" * 64)

    with pytest.raises(ValueError, match="different ICLabel models"):
        compare_iclabel_reports(native, pyodide)


def test_compare_iclabel_reports_rejects_swapped_platform_labels():
    native = _iclabel_report("native", _two_component_values())
    pyodide = _iclabel_report("pyodide", _two_component_values())

    with pytest.raises(ValueError, match="labeled native and pyodide"):
        compare_iclabel_reports(pyodide, native)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 1}, "Unsupported ICLabel parity schema"),
        ({"platform": "browser"}, "Invalid ICLabel parity platform"),
        ({"model_sha256": None}, "lacks a SHA-256 model digest"),
        ({"model_sha256": "not-a-digest"}, "lacks a SHA-256 model digest"),
        ({"shape": [3, 7]}, "does not match classifications"),
        ({"classifications": [[0.5] * 6, [0.5] * 6], "shape": [2, 6]}, "Unexpected ICLabel classification shape"),
        ({"classifications": [[0.1] * 7, [float("nan")] * 7]}, "non-finite values"),
        ({"classifications": [[0.1] * 7, [float("inf")] * 7]}, "non-finite values"),
    ],
)
def test_load_iclabel_report_rejects_invalid_reports(tmp_path, overrides, message):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({**_iclabel_report("native", _two_component_values()), **overrides}))

    with pytest.raises(ValueError, match=message):
        compare_iclabel._load_report(path)


def test_native_parity_report_round_trips_through_the_comparison(tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    native = run_parity("native", REPO_ROOT / "sample_data")
    packaged_model = REPO_ROOT / "src" / "eegprep" / "plugins" / "ICLabel" / "iclabel.onnx"

    assert native["model_sha256"] == hashlib.sha256(packaged_model.read_bytes()).hexdigest()
    assert native["shape"] == [32, 7]

    arguments = _write_reports(tmp_path, native, {**native, "platform": "pyodide"})
    assert compare_iclabel.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["max_absolute_difference"] == 0.0


def test_browser_backend_without_a_host_names_the_bridge_contract(monkeypatch):
    image = np.zeros((1, 1, 1, 1), dtype=np.float32)

    # Outside Pyodide there is no ``js`` module at all.
    monkeypatch.setitem(sys.modules, "js", None)
    with pytest.raises(ImportError, match="createIcLabelWebBridge"):
        asyncio.run(run_iclabel_net_async(image, image, image))

    # Inside Pyodide, a host that never registered the global leaves ``js`` without it.
    monkeypatch.setitem(sys.modules, "js", types.ModuleType("js"))
    with pytest.raises(ImportError, match="eegprep_iclabel_web"):
        asyncio.run(run_iclabel_net_async(image, image, image))


@pytest.mark.parametrize(
    ("name", "status", "in_lock", "wheel"),
    [
        ("numpy", "ok", True, None),
        ("mne", "ok", False, "mne-1.10.0-py3-none-any.whl"),
        ("docopt", "ok", False, None),
        ("no-wasm-build", "FAIL", False, None),
    ],
)
def test_base_resolution_classifies_each_package_once(name, status, in_lock, wheel):
    result = check_pyodide_base_resolution.classify_package(name, "1.0", in_lock, wheel)

    assert result.status == status


def test_base_resolution_reports_declared_known_gaps(monkeypatch):
    monkeypatch.setitem(check_pyodide_base_resolution.KNOWN_GAPS, "gap-package", "no wasm build yet")

    result = check_pyodide_base_resolution.classify_package("gap-package", "1.0", False, None)

    assert result.status == "known-gap"
    assert "no wasm build yet" in result.source


def test_pinned_pyodide_version_agrees_across_the_harness():
    package = json.loads((REPO_ROOT / "tools" / "pyodide" / "package.json").read_text())
    version = package["dependencies"]["pyodide"]

    assert version == check_pyodide_base_resolution.PYODIDE_VERSION
    assert f"Pyodide {version}" in (REPO_ROOT / "docs" / "source" / "pyodide_benchmark.md").read_text()


def test_harness_holds_mne_at_the_declared_floor():
    dependencies = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    (floor,) = [item.removeprefix("mne>=") for item in dependencies if item.startswith("mne>=")]
    runner = (REPO_ROOT / "tools" / "pyodide" / "run_pyodide.mjs").read_text()

    assert re.search(r"MICROPIP_CONSTRAINTS = \[[^\]]*\"mne==" + re.escape(floor) + r"\"", runner)


def _run_script_with_path(path_entries, tmp_path):
    """Run run_pyodide.sh with a PATH of only the given directories, after the argument checks."""
    wheel = tmp_path / "eegprep.whl"
    script = tmp_path / "script.py"
    wheel.touch()
    script.touch()
    return subprocess.run(
        [
            shutil.which("bash"),
            str(REPO_ROOT / "tools" / "pyodide" / "run_pyodide.sh"),
            "--wheel",
            str(wheel),
            "--script",
            str(script),
        ],
        env={"PATH": ":".join(str(entry) for entry in path_entries), "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to run run_pyodide.sh")
@pytest.mark.parametrize("missing", ["bun", "node"])
def test_runner_script_reports_a_missing_javascript_tool_before_downloading(tmp_path, missing):
    # Only the helpers the script needs before its checks, plus a bun or node stub on
    # PATH: the script must stop with exit code 2 before running any installer.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for helper in ("mktemp", "dirname", "rm"):
        (bin_dir / helper).symlink_to(shutil.which(helper))
    installer_log = tmp_path / "bun-was-run"
    present = "node" if missing == "bun" else "bun"
    stub = bin_dir / present
    stub.write_text(f"#!/bin/sh\ntouch {installer_log}\n")
    stub.chmod(0o755)

    result = _run_script_with_path([bin_dir], tmp_path)

    assert result.returncode == 2
    assert missing in result.stderr
    assert not installer_log.exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the browser bridge test")
def test_iclabel_web_bridge_retries_failed_session_initialization():
    script = textwrap.dedent(
        """
        import { createIcLabelWebBridge } from './tools/pyodide/iclabel_web_bridge.mjs';

        let attempts = 0;
        const ort = {
          InferenceSession: {
            create: async () => {
              attempts += 1;
              if (attempts === 1) throw new Error('transient model load failure');
              return { run: async () => ({ output: { data: [1] } }) };
            },
          },
          Tensor: class Tensor {
            constructor(type, data, shape) {
              this.type = type;
              this.data = data;
              this.shape = shape;
            }
          },
        };

        const bridge = createIcLabelWebBridge(ort, new Uint8Array());
        const input = new Float32Array([0]);
        const shape = [1];
        try {
          await bridge.run(input, shape, input, shape, input, shape);
          throw new Error('first run unexpectedly succeeded');
        } catch (error) {
          if (error.message !== 'transient model load failure') throw error;
        }
        await bridge.run(input, shape, input, shape, input, shape);
        if (attempts !== 2) throw new Error(`expected two session attempts, got ${attempts}`);
        """
    )
    result = subprocess.run(
        ["node", "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        cwd=Path(__file__).parents[1],
        check=False,
    )
    assert result.returncode == 0, result.stderr
