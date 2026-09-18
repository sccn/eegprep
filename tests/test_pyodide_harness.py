import pytest

from tools.check_pyodide_base_resolution import KNOWN_GAPS
from tools.pyodide.benchmark import MATMUL_CASES, MAX_ICA_ITERATIONS, benchmark_record
from tools.pyodide.compare_benchmarks import compare_reports
from tools.pyodide.compare_iclabel import compare_reports as compare_iclabel_reports
from tools.pyodide.prepare_docopt_wheel import verify_sha256


def test_docopt_hash_check_rejects_modified_sdist(tmp_path):
    archive = tmp_path / "docopt-0.6.2.tar.gz"
    archive.write_bytes(b"locked sdist payload")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(archive, "0" * 64)


def test_docopt_is_no_longer_a_static_known_gap():
    assert "docopt" not in KNOWN_GAPS


def test_benchmark_record_marks_max_iterations_as_not_converged():
    result = benchmark_record(
        algorithm="runica",
        platform="native",
        shape=(64, 15_000),
        seed=375,
        seconds=1.25,
        iterations=MAX_ICA_ITERATIONS,
        converged=False,
        backend="numpy",
    )

    assert result["iterations"] == MAX_ICA_ITERATIONS
    assert result["converged"] is False
    assert result["error"] is None


def test_benchmark_uses_runica_inner_loop_product_shapes():
    assert [(case["left_shape"], case["right_shape"]) for case in MATMUL_CASES] == [
        ((64, 64), (64, 49)),
        ((64, 64), (64, 64)),
    ]


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
    assert comparison["decisions"]["phase3_recommended"] is False


def test_compare_iclabel_reports_applies_established_numeric_tolerance():
    native = {
        "schema_version": 1,
        "platform": "native",
        "dataset": "eeglab_data_with_ica_tmp.set",
        "shape": [2, 7],
        "classifications": [[0.1] * 7, [0.9] * 7],
    }
    pyodide = {
        **native,
        "platform": "pyodide",
        "classifications": [[0.1 + 1e-6] * 7, [0.9 - 1e-6] * 7],
    }

    comparison = compare_iclabel_reports(native, pyodide)

    assert comparison["allclose"] is True
    assert comparison["max_absolute_difference"] <= 1e-5


def test_compare_iclabel_reports_rejects_shape_mismatch():
    native = {
        "schema_version": 1,
        "platform": "native",
        "dataset": "sample.set",
        "shape": [1, 7],
        "classifications": [[0.1] * 7],
    }
    pyodide = {
        **native,
        "platform": "pyodide",
        "shape": [2, 7],
        "classifications": [[0.1] * 7, [0.9] * 7],
    }

    with pytest.raises(ValueError, match="shapes differ"):
        compare_iclabel_reports(native, pyodide)
