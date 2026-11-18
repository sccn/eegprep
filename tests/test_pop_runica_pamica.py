"""``pop_runica(icatype='pamica')``: EEGLAB ICA fields, run record, and the E1 comparison with AMICA.

Rule E1: on ``sample_data/eeglab_data.set``, with one model, three mixture components, 2000 iterations,
float64 and one thread, the Hungarian-matched absolute correlations between the rows of the sensor-space
unmixing matrices of ``pamica`` and ``runamica15`` are compared with a minimum of 0.95 and a median of
0.99, and their Amari distance with 0.05. The comparison is reported, not asserted: see the E1 test.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pamica
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_info

from eegprep import pop_loadset
from eegprep.functions.popfunc import eeg_pamica as pamica_module
from eegprep.functions.popfunc.eeg_pamica import (
    PamicaFitError,
    _limited_threads,
    _require_thread_count,
    eeg_pamica,
)
from eegprep.functions.popfunc.pop_runica import pop_runica
from eegprep.functions.sigprocfunc.runamica import is_amica_available
from tests.fixtures import create_test_eeg
from tests.test_ds003061_stage_parity import amari_distance

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"
RESULT_KEYS = {
    "version",
    "torch",
    "device",
    "dtype",
    "threads",
    "rank",
    "backend",
    "pipeline",
    "iterations",
    "stop_reason",
    "did_converge",
    "final_ll",
    "best_ll",
    "best_iter",
}
E1_MIN_CORRELATION = 0.95
E1_MEDIAN_CORRELATION = 0.99
E1_MAX_AMARI = 0.05


def _synthetic_eeg(n_channels=6, n_samples=3000):
    generator = np.random.RandomState(11)
    sources = np.vstack(
        [
            generator.laplace(size=n_samples) if index % 2 else generator.uniform(-1, 1, n_samples)
            for index in range(n_channels)
        ]
    )
    eeg = create_test_eeg(n_channels=n_channels, n_samples=n_samples)
    eeg["data"] = generator.randn(n_channels, n_channels) @ sources
    return eeg


def _replay(eeg, record):
    """Fit pAMICA from an ``etc['pamica']`` record under that record's thread count."""
    settings = {key: value for key, value in record.items() if key not in RESULT_KEYS}
    with _limited_threads(torch, record["threads"]):
        model = pamica.AMICA(
            n_models=settings.pop("n_models"), n_mix=settings.pop("n_mix"), device="cpu", verbose=False
        )
        return model.fit(
            np.asarray(eeg["data"], dtype=np.float64),
            max_iter=settings.pop("max_iter"),
            lrate=settings.pop("lrate"),
            do_mean=settings.pop("do_mean"),
            do_sphere=settings.pop("do_sphere"),
            do_newton=settings.pop("do_newton"),
            dtype=torch.float64,
            **settings,
        )


def test_icawinv_times_icaact_reconstructs_the_data_at_full_rank():
    eeg = _synthetic_eeg()

    result = pop_runica(eeg, icatype="pamica", max_iter=40)

    data = np.asarray(eeg["data"], dtype=np.float64)
    reconstructed = result["icawinv"] @ result["icaact"]
    assert np.max(np.abs(reconstructed - data)) <= 1e-8 * np.max(np.abs(data))
    assert result["icaweights"].shape == (6, 6)
    assert result["icasphere"].shape == (6, 6)
    assert np.array_equal(result["icachansind"], np.arange(6))
    assert eeg["icaweights"] is None


@pytest.mark.parametrize("seed", [None, 5489])
def test_same_seed_gives_bit_identical_weights(seed):
    eeg = _synthetic_eeg()
    options = {} if seed is None else {"seed": seed}

    first = pop_runica(eeg, icatype="pamica", max_iter=40, **options)
    second = pop_runica(eeg, icatype="pamica", max_iter=40, **options)

    assert np.array_equal(first["icaweights"], second["icaweights"])
    assert np.array_equal(first["icasphere"], second["icasphere"])


def test_another_seed_starts_elsewhere():
    eeg = _synthetic_eeg()

    first = pop_runica(eeg, icatype="pamica", max_iter=3, seed=5489)
    other = pop_runica(eeg, icatype="pamica", max_iter=3, seed=1)

    assert not np.array_equal(first["icaweights"], other["icaweights"])


def test_run_record_names_the_run_and_replays_it():
    eeg = _synthetic_eeg()

    result = eeg_pamica(eeg, max_iter=40)

    record = result["etc"]["pamica"]
    assert record["version"] == pamica.__version__
    assert record["torch"] == torch.__version__
    assert (record["device"], record["dtype"], record["threads"]) == ("cpu", "float64", 1)
    assert (record["seed"], record["n_models"], record["n_mix"]) == (5489, 1, 3)
    assert (record["max_iter"], record["block_size"], record["pcakeep"]) == (40, 131072, None)
    assert {key: record[key] for key in ("lrate", "minlrate", "rho0", "newt_start")} == {
        "lrate": 0.05,
        "minlrate": 1e-8,
        "rho0": 1.5,
        "newt_start": 50,
    }
    assert [record[key] for key in ("do_newton", "use_min_dll", "use_grad_norm", "do_mean", "do_sphere")] == [
        True,
        False,
        False,
        True,
        True,
    ]
    assert {key: record[key] for key in ("maxdecs", "newtrate", "rholratefact", "invsigmin", "invsigmax")} == {
        "maxdecs": 3,
        "newtrate": 1.0,
        "rholratefact": 0.5,
        "invsigmin": 1e-5,
        "invsigmax": 100.0,
    }
    assert record["iterations"] == 40
    assert 0 <= record["best_iter"] < record["iterations"]
    model = _replay(eeg, record)
    assert np.array_equal(model.get_unmixing_matrix(), result["icaweights"])
    assert np.array_equal(model.get_sphere(), result["icasphere"])
    assert model.final_ll_ == record["final_ll"]
    assert len(model.ll_history_) == record["iterations"]
    assert max(model.ll_history_) == record["best_ll"]


def test_the_record_names_the_pamica_pipeline_and_is_enough_to_reproduce_the_run():
    eeg = _synthetic_eeg()

    result = eeg_pamica(eeg, max_iter=40, newt_start=3)

    record = result["etc"]["pamica"]
    assert record["pipeline"] == "pamica.AMICA(backend='torch').fit"
    assert record["backend"] == "AMICATorchNG"
    first = _replay(eeg, record)
    second = _replay(eeg, record)
    assert type(first.model_).__name__ == record["backend"]
    assert first.backend == "torch"
    assert np.array_equal(first.get_unmixing_matrix(), second.get_unmixing_matrix())
    assert np.array_equal(first.get_unmixing_matrix(), result["icaweights"])
    assert np.array_equal(first.get_sphere(), result["icasphere"])


def test_a_caller_value_replaces_the_eeglab_default_and_is_recorded():
    eeg = _synthetic_eeg()

    result = eeg_pamica(eeg, n_mix=2, max_iter=10, block_size=500, seed=7, lrate=0.02, newt_start=3, do_newton=False)

    record = result["etc"]["pamica"]
    assert (record["n_mix"], record["max_iter"], record["block_size"], record["seed"]) == (2, 10, 500, 7)
    assert (record["lrate"], record["newt_start"], record["do_newton"]) == (0.02, 3, False)
    assert np.array_equal(_replay(eeg, record).get_unmixing_matrix(), result["icaweights"])


def test_eegprep_amica_option_names_reach_pamica_under_its_names():
    eeg = _synthetic_eeg()
    settings = {"max_iter": 30, "use_grad_norm": True}

    amica_names = pop_runica(eeg, icatype="pamica", max_decs=2, min_grad_norm=1e9, **settings)
    pamica_names = pop_runica(eeg, icatype="pamica", maxdecs=2, min_nd=1e9, **settings)
    by_text = pop_runica(
        eeg, icatype="pamica", options="'max_iter', 30, 'use_grad_norm', 1, 'max_decs', 2, 'min_grad_norm', 1e9"
    )

    for result in (amica_names, pamica_names, by_text):
        record = result["etc"]["pamica"]
        assert (record["maxdecs"], record["min_nd"]) == (2, 1e9)
        assert "max_decs" not in record
        assert "min_grad_norm" not in record
        assert record["stop_reason"] == "grad_norm"
    assert np.array_equal(amica_names["icaweights"], pamica_names["icaweights"])
    assert np.array_equal(by_text["icaweights"], pamica_names["icaweights"])


def test_eeglab_option_names_reach_pamica_under_its_names():
    eeg = _synthetic_eeg()

    by_keyword = pop_runica(eeg, icatype="pamica", maxiter=5, num_mix_comps=2, num_models=1)
    by_text = pop_runica(eeg, icatype="pamica", options="'maxiter', 5, 'num_mix_comps', 2, 'num_models', 1")

    for result in (by_keyword, by_text):
        record = result["etc"]["pamica"]
        assert (record["max_iter"], record["n_mix"], record["n_models"]) == (5, 2, 1)
    assert np.array_equal(by_keyword["icaweights"], by_text["icaweights"])


def test_icaweights_and_icasphere_are_the_unmixing_pamica_applies():
    eeg = _synthetic_eeg()
    data = np.asarray(eeg["data"], dtype=np.float64)

    result = eeg_pamica(eeg, max_iter=40)

    sources = _replay(eeg, result["etc"]["pamica"]).transform(data)
    linear = result["icaweights"] @ result["icasphere"] @ data
    # pAMICA subtracts the channel mean before sphering and EEGLAB's activations do not, so the two differ
    # by one constant per component.
    offset = sources - linear
    constant = np.abs(offset - offset.mean(axis=1, keepdims=True)).max()
    assert constant <= 1e-10 * np.abs(sources).max()
    assert np.allclose(result["icaact"], linear, rtol=0, atol=1e-12 * np.abs(linear).max())
    assert np.allclose(
        result["icawinv"], np.linalg.pinv(result["icaweights"] @ result["icasphere"]), rtol=1e-8, atol=1e-10
    )


def test_without_mean_removal_transform_equals_the_linear_unmixing():
    eeg = _synthetic_eeg()
    data = np.asarray(eeg["data"], dtype=np.float64)

    result = eeg_pamica(eeg, max_iter=40, do_mean=False)

    sources = _replay(eeg, result["etc"]["pamica"]).transform(data)
    linear = result["icaweights"] @ result["icasphere"] @ data
    assert np.abs(sources - linear).max() <= 1e-10 * np.abs(sources).max()


def test_without_sphering_icasphere_is_the_identity():
    eeg = _synthetic_eeg()

    result = eeg_pamica(eeg, max_iter=10, do_sphere=False)

    assert np.array_equal(result["icasphere"], np.eye(6))


def test_the_record_gives_the_rank_that_ran_beside_the_requested_pcakeep():
    eeg = _synthetic_eeg()

    sphered = eeg_pamica(eeg, max_iter=10, pcakeep=4)
    unsphered = eeg_pamica(eeg, max_iter=10, pcakeep=4, do_sphere=False)

    # pAMICA ignores pcakeep when it does not sphere, so the six components that ran are not the four requested.
    assert (sphered["etc"]["pamica"]["pcakeep"], sphered["etc"]["pamica"]["rank"]) == (4, 4)
    assert (unsphered["etc"]["pamica"]["pcakeep"], unsphered["etc"]["pamica"]["rank"]) == (4, 6)
    assert unsphered["icaweights"].shape[0] == 6
    assert eeg_pamica(eeg, max_iter=10)["etc"]["pamica"]["rank"] == 6


def test_rank_reduced_fit_maps_components_to_all_channels():
    eeg = _synthetic_eeg()
    data = np.asarray(eeg["data"], dtype=np.float64)

    result = pop_runica(eeg, icatype="pamica", max_iter=40, pcakeep=4, reorder="off")

    assert result["icaweights"].shape == (4, 4)
    assert result["icasphere"].shape == (4, 6)
    assert result["icawinv"].shape == (6, 4)
    assert np.array_equal(result["icachansind"], np.arange(6))
    assert result["etc"]["pamica"]["pcakeep"] == 4
    model = _replay(eeg, result["etc"]["pamica"])
    assert np.array_equal(model.get_unmixing_matrix(), result["icaweights"])
    unmixing = result["icaweights"] @ result["icasphere"]
    assert np.allclose(result["icawinv"], np.linalg.pinv(unmixing), rtol=1e-8, atol=1e-10)
    assert np.allclose(result["icaact"], unmixing @ data, rtol=0, atol=1e-12 * np.abs(unmixing @ data).max())
    assert np.allclose(unmixing @ result["icawinv"], np.eye(4), atol=1e-8)


def test_reorder_sorts_components_by_variance_and_off_keeps_pamica_order():
    eeg = _synthetic_eeg()

    kept = pop_runica(eeg, icatype="pamica", max_iter=40, reorder="off")
    sorted_result = pop_runica(eeg, icatype="pamica", max_iter=40, reorder="on")

    variance = np.sum(sorted_result["icawinv"] ** 2, axis=0) * np.sum(sorted_result["icaact"] ** 2, axis=1)
    assert np.all(np.diff(variance) <= 0)
    assert np.array_equal(kept["icaweights"], _replay(eeg, kept["etc"]["pamica"]).get_unmixing_matrix())
    # pop_runica sorts by default; eeg_pamica called directly keeps pAMICA's order by default.
    assert np.array_equal(pop_runica(eeg, icatype="pamica", max_iter=40)["icaweights"], sorted_result["icaweights"])
    assert np.array_equal(eeg_pamica(eeg, max_iter=40)["icaweights"], kept["icaweights"])
    assert sorted(map(tuple, np.round(kept["icaweights"], 12))) == sorted(
        map(tuple, np.round(sorted_result["icaweights"], 12))
    )


def test_a_fit_that_stops_on_max_iter_has_not_converged():
    eeg = _synthetic_eeg()

    result = eeg_pamica(eeg, max_iter=5)

    record = result["etc"]["pamica"]
    assert record["stop_reason"] == "max_iter"
    assert record["did_converge"] is False
    assert record["iterations"] == 5


@pytest.mark.parametrize(
    ("stop_reason", "options"),
    [
        ("grad_norm", {"use_grad_norm": True, "min_nd": 1e9}),
        ("min_dll", {"use_min_dll": True, "min_dll": 1e3}),
        ("lrate_floor", {"lrate": 3.0, "minlrate": 1.0}),
        ("grad_norm_floor", {"lrate": 3.0, "min_nd": 1e9}),
    ],
)
def test_a_fit_that_stops_on_a_convergence_check_has_converged(stop_reason, options):
    eeg = _synthetic_eeg()

    result = eeg_pamica(eeg, max_iter=500, **options)

    record = result["etc"]["pamica"]
    assert record["stop_reason"] == stop_reason
    assert record["did_converge"] is True
    assert record["iterations"] < 500


def test_a_degenerate_fit_raises_and_sets_no_ica_field():
    eeg = _synthetic_eeg()

    with pytest.raises(PamicaFitError, match="nan_params"):
        pop_runica(eeg, icatype="pamica", max_iter=10, lrate=float("nan"))
    with pytest.raises(PamicaFitError, match="nan_params"):
        eeg_pamica(eeg, max_iter=10, lrate=float("nan"))

    assert eeg["icaweights"] is None
    assert eeg["icasphere"] is None
    assert eeg["etc"] == {}


def test_newton_changes_the_fit_once_it_starts():
    eeg = _synthetic_eeg()

    newton = eeg_pamica(eeg, max_iter=20, newt_start=3, do_newton=True)
    plain = eeg_pamica(eeg, max_iter=20, newt_start=3, do_newton=False)

    assert (newton["etc"]["pamica"]["do_newton"], plain["etc"]["pamica"]["do_newton"]) == (True, False)
    assert np.abs(newton["icaweights"] - plain["icaweights"]).max() > 0.1


def test_newtrate_changes_the_newton_fit():
    eeg = _synthetic_eeg()

    fast = eeg_pamica(eeg, max_iter=20, newt_start=3, newtrate=1.0)
    slow = eeg_pamica(eeg, max_iter=20, newt_start=3, newtrate=0.3)

    assert (fast["etc"]["pamica"]["newtrate"], slow["etc"]["pamica"]["newtrate"]) == (1.0, 0.3)
    assert np.abs(fast["icaweights"] - slow["icaweights"]).max() > 0.1


def test_a_newton_fit_repeats_and_replays_bit_for_bit():
    eeg = _synthetic_eeg()

    first = eeg_pamica(eeg, max_iter=20, newt_start=3, newtrate=0.3)
    second = eeg_pamica(eeg, max_iter=20, newt_start=3, newtrate=0.3)

    assert np.array_equal(first["icaweights"], second["icaweights"])
    assert np.array_equal(_replay(eeg, first["etc"]["pamica"]).get_unmixing_matrix(), first["icaweights"])


def test_two_models_fit_differently_from_one():
    eeg = _synthetic_eeg()

    one = eeg_pamica(eeg, max_iter=20)
    two = eeg_pamica(eeg, max_iter=20, n_models=2)

    assert (one["etc"]["pamica"]["n_models"], two["etc"]["pamica"]["n_models"]) == (1, 2)
    assert np.abs(one["icaweights"] - two["icaweights"]).max() > 1e-3
    assert np.array_equal(_replay(eeg, two["etc"]["pamica"]).get_unmixing_matrix(), two["icaweights"])


def test_the_input_dataset_is_left_unchanged():
    eeg = _synthetic_eeg()
    before = copy.deepcopy(eeg)

    result = eeg_pamica(eeg, max_iter=5)

    assert result is not eeg
    assert np.array_equal(eeg["data"], before["data"])
    assert eeg["icaweights"] is None
    assert eeg["icasphere"] is None
    assert eeg["icaact"] is None
    assert eeg["etc"] == {}


def test_a_failure_outside_the_import_guard_propagates_unchanged():
    eeg = _synthetic_eeg()
    eeg["data"][2, 5] = np.nan

    with pytest.raises(torch.linalg.LinAlgError) as caught:
        eeg_pamica(eeg, max_iter=5)

    assert caught.value.__cause__ is None
    assert eeg["icaweights"] is None


def test_the_activations_are_computed_without_numpy_matmul_warnings():
    eeg = _synthetic_eeg()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = eeg_pamica(eeg, max_iter=5)

    assert np.isfinite(result["icaact"]).all()


def test_an_epoched_dataset_gives_epoched_activations():
    generator = np.random.RandomState(5)
    channels, points, trials = 4, 500, 6
    sources = np.vstack(
        [
            generator.laplace(size=points * trials) if index % 2 else generator.uniform(-1, 1, points * trials)
            for index in range(channels)
        ]
    )
    eeg = create_test_eeg(n_channels=channels, n_samples=points, n_trials=trials)
    eeg["data"] = (generator.randn(channels, channels) @ sources).reshape(channels, points, trials, order="F")

    result = pop_runica(eeg, icatype="pamica", max_iter=30)

    assert result["icaact"].shape == (channels, points, trials)
    flat = eeg["data"].reshape(channels, -1, order="F")
    reconstructed = result["icawinv"] @ result["icaact"].reshape(channels, -1, order="F")
    assert np.max(np.abs(reconstructed - flat)) <= 1e-8 * np.max(np.abs(flat))


def _high_thread_environment_replay():
    """Fit and replay once in a process whose BLAS and OpenMP pools were started large."""
    assert os.environ["OMP_NUM_THREADS"] == "32"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "32"
    pools = [pool for pool in threadpool_info() if pool["user_api"] in ("blas", "openmp")]
    assert torch.get_num_threads() > 1 or any(pool["num_threads"] > 1 for pool in pools)
    eeg = _synthetic_eeg()

    result = eeg_pamica(eeg, max_iter=40)

    assert result["etc"]["pamica"]["threads"] == 1
    weights = _replay(eeg, result["etc"]["pamica"]).get_unmixing_matrix()
    if not np.array_equal(weights, result["icaweights"]):
        delta = np.abs(weights - result["icaweights"])
        scale = np.maximum(np.abs(weights), np.abs(result["icaweights"]))
        raise AssertionError(f"max abs {delta.max()} max rel {(delta / np.maximum(scale, 1e-300)).max()}")


def _two_thread_counts_in_one_process():
    """Fit twice at different thread counts, then replay each result."""
    eeg = _synthetic_eeg()

    one = eeg_pamica(eeg, max_iter=20, threads=1)
    four = pop_runica(eeg, icatype="pamica", max_iter=20, threads=4, reorder="off")
    again = eeg_pamica(eeg, max_iter=20, threads=1)

    assert (one["etc"]["pamica"]["threads"], four["etc"]["pamica"]["threads"]) == (1, 4)
    assert np.array_equal(_replay(eeg, one["etc"]["pamica"]).get_unmixing_matrix(), one["icaweights"])
    assert np.array_equal(_replay(eeg, four["etc"]["pamica"]).get_unmixing_matrix(), four["icaweights"])
    assert np.array_equal(again["icaweights"], one["icaweights"])


def test_two_thread_counts_in_one_process_replay_exactly():
    root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_pop_runica_pamica import _two_thread_counts_in_one_process; "
            "_two_thread_counts_in_one_process()",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr


def test_high_environment_thread_pools_still_replay_exactly():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "32"
    env["OPENBLAS_NUM_THREADS"] = "32"
    env["MKL_NUM_THREADS"] = "32"
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_pop_runica_pamica import _high_thread_environment_replay; "
            "_high_thread_environment_replay()",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr


class _OpenMPPool:
    """One loaded OpenMP library. ``accepts`` is false when ``omp_set_num_threads`` does nothing."""

    def __init__(self, filepath, count, *, accepts=True):
        self.filepath = filepath
        self.count = count
        self.accepts = accepts
        self.sets = []

    def get_num_threads(self):
        return self.count

    def set_num_threads(self, count):
        self.sets.append(count)
        if self.accepts:
            self.count = count


class _OpenMPController:
    def __init__(self, pools):
        self.lib_controllers = pools

    def select(self, user_api):
        assert user_api == "openmp"
        return self


class _TorchThreads:
    def get_num_threads(self):
        return 4


def test_a_real_openmp_pool_stuck_at_one_still_rejects_the_count(monkeypatch):
    stub = _OpenMPPool(r"D:\venv\Lib\site-packages\torch\lib\libiompstubs5md.dll", 1, accepts=False)
    real = _OpenMPPool(r"D:\venv\Lib\site-packages\torch\lib\libiomp5md.dll", 1, accepts=True)
    monkeypatch.setattr(
        pamica_module,
        "threadpool_info",
        lambda: [{"user_api": "blas", "internal_api": "openblas", "num_threads": 4}],
    )
    monkeypatch.setattr(pamica_module, "ThreadpoolController", lambda: _OpenMPController([stub, real]))

    with pytest.raises(RuntimeError, match=r"openmp/libiomp5md\.dll is 1") as caught:
        _require_thread_count(_TorchThreads(), 4)

    assert "libiompstubs" not in str(caught.value)
    assert real.count == 1
    assert real.sets == []
    assert stub.sets == []


def test_threads_are_limited_for_the_run_and_restored_after():
    eeg = _synthetic_eeg()
    torch_before = torch.get_num_threads()
    torch.set_num_threads(2)
    pools_before = [(pool["internal_api"], pool["num_threads"]) for pool in threadpool_info()]
    try:
        result = eeg_pamica(eeg, max_iter=3, threads=1)

        assert result["etc"]["pamica"]["threads"] == 1
        assert torch.get_num_threads() == 2
        assert [(pool["internal_api"], pool["num_threads"]) for pool in threadpool_info()] == pools_before
    finally:
        torch.set_num_threads(torch_before)


@pytest.mark.parametrize("missing", ["pamica", "torch"])
def test_missing_library_names_the_pamica_extra(monkeypatch, missing):
    eeg = _synthetic_eeg()
    monkeypatch.setitem(sys.modules, missing, None)

    with pytest.raises(ImportError, match="eegprep\\[pamica\\]") as caught:
        pop_runica(eeg, icatype="pamica", max_iter=3)

    assert missing in str(caught.value)
    assert isinstance(caught.value.__cause__, ImportError)
    assert caught.value.__cause__.name == missing


def test_pop_runica_history_names_the_backend():
    eeg = _synthetic_eeg()

    _result, command = pop_runica(eeg, icatype="pamica", max_iter=3, seed=7, return_com=True)

    assert command == "EEG = pop_runica(EEG, 'icatype', 'pamica', 'max_iter', 3, 'seed', 7);"


def _matched_correlations(left, right):
    """Absolute correlations of the Hungarian-matched rows of two unmixing matrices."""
    count = left.shape[0]
    correlation = np.abs(np.corrcoef(left, right)[:count, count:])
    rows, columns = linear_sum_assignment(-correlation)
    return correlation[rows, columns]


def _e1_statistics(pamica_unmixing, amica_unmixing):
    matched = _matched_correlations(pamica_unmixing, amica_unmixing)
    amari = amari_distance(pamica_unmixing, np.linalg.pinv(amica_unmixing))
    return float(matched.min()), float(np.median(matched)), float(amari)


def _assert_e1(minimum, median, amari):
    assert minimum >= E1_MIN_CORRELATION, f"E1 minimum matched correlation {minimum:.4f} < {E1_MIN_CORRELATION}"
    assert median >= E1_MEDIAN_CORRELATION, f"E1 median matched correlation {median:.4f} < {E1_MEDIAN_CORRELATION}"
    assert amari <= E1_MAX_AMARI, f"E1 Amari distance {amari:.4f} > {E1_MAX_AMARI}"


@pytest.mark.parametrize(
    ("statistics", "message"),
    [((0.949, 0.995, 0.01), "minimum"), ((0.96, 0.989, 0.01), "median"), ((0.96, 0.995, 0.051), "Amari")],
)
def test_e1_thresholds_are_the_pre_registered_numbers(statistics, message):
    _assert_e1(0.95, 0.99, 0.05)

    with pytest.raises(AssertionError, match=message):
        _assert_e1(*statistics)


def test_e1_statistics_report_the_known_matched_correlations():
    generator = np.random.RandomState(2)
    basis = generator.randn(10, 6)
    basis -= basis.mean(axis=0, keepdims=True)
    orthonormal, _ = np.linalg.qr(basis)
    left, other = orthonormal[:, :3].T, orthonormal[:, 3:].T
    cosines = np.array([1.0, 0.98, 0.9])
    right = cosines[:, None] * left + np.sqrt(1 - cosines**2)[:, None] * other

    minimum, median, _amari = _e1_statistics(left, right[[2, 0, 1]])

    assert minimum == pytest.approx(0.9)
    assert median == pytest.approx(0.98)


def test_e1_statistics_pass_a_signed_permutation_and_fail_a_wrong_pairing():
    generator = np.random.RandomState(0)
    unmixing = generator.randn(12, 12)
    signs = np.diag(generator.choice([-1.0, 1.0], size=12))
    permuted = signs @ np.eye(12)[generator.permutation(12)] @ unmixing

    _assert_e1(*_e1_statistics(unmixing, permuted))

    wrong = permuted.copy()
    wrong[[3, 7]] = wrong[[7, 3]]
    wrong[3] = wrong[7]
    with pytest.raises(AssertionError, match="E1"):
        _assert_e1(*_e1_statistics(unmixing, wrong))


@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_e1_reports_pamica_against_amica_on_the_sample_recording(capsys):
    """Print how pamica compares with EEGLAB's AMICA binary on the sample recording; assert nothing about it.

    E1 compares the Hungarian-matched absolute correlations of the sensor-space unmixing rows and the
    Amari distance of ``pamica`` and ``runamica15``. The reference values 0.95, 0.99 and 0.05 are printed
    beside the measured ones so the verdict can be read by eye. The numbers are informational because the
    shipped AMICA binary seeds its random start from the clock: two runs of this code gave a minimum
    matched correlation of 0.9966 and of 0.5339. pamica itself is deterministic for a fixed seed. The test
    still runs both engines, skips with a reason when the binary cannot run, and fails instead of skipping
    when ``EEGPREP_REQUIRE_AMICA=1``.
    """
    if not is_amica_available():
        if os.environ.get("EEGPREP_REQUIRE_AMICA") == "1":
            pytest.fail("EEGPREP_REQUIRE_AMICA=1 but the shipped AMICA binary cannot run on this host")
        pytest.skip("the shipped AMICA binary cannot run on this host")
    eeg = pop_loadset(str(SAMPLE))

    started = time.perf_counter()
    pamica_eeg = pop_runica(eeg, icatype="pamica", max_iter=2000, reorder="off")
    pamica_seconds = time.perf_counter() - started
    started = time.perf_counter()
    amica_eeg = pop_runica(eeg, icatype="runamica15", maxiter=2000, max_threads=1, reorder="off")
    amica_seconds = time.perf_counter() - started

    minimum, median, amari = _e1_statistics(
        pamica_eeg["icaweights"] @ pamica_eeg["icasphere"], amica_eeg["icaweights"] @ amica_eeg["icasphere"]
    )
    with capsys.disabled():
        print(
            f"\nE1 min={minimum:.4f} median={median:.4f} amari={amari:.4f} "
            f"pamica_s={pamica_seconds:.1f} amica_s={amica_seconds:.1f} "
            f"| reference min>={E1_MIN_CORRELATION} median>={E1_MEDIAN_CORRELATION} amari<={E1_MAX_AMARI}"
        )
