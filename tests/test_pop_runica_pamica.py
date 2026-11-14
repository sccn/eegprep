"""``pop_runica(icatype='pamica')``: EEGLAB ICA fields, run record, and rule E1 against AMICA.

Rule E1 (pre-registered, never loosened): on ``sample_data/eeglab_data.set``, with one model, three
mixture components, 2000 iterations, float64 and one thread, the Hungarian-matched absolute
correlations between the rows of the sensor-space unmixing matrices of ``pamica`` and ``runamica15``
have minimum at least 0.95 and median at least 0.99, and their Amari distance is at most 0.05.
"""

from __future__ import annotations

import copy
import os
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
from eegprep.functions.popfunc.eeg_pamica import PamicaFitError, eeg_pamica
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
    """Fit pAMICA directly from an ``etc['pamica']`` record and return the fitted model."""
    settings = {key: value for key, value in record.items() if key not in RESULT_KEYS}
    model = pamica.AMICA(n_models=settings.pop("n_models"), n_mix=settings.pop("n_mix"), device="cpu", verbose=False)
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
def test_e1_pamica_matches_amica_on_the_sample_recording(capsys):
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
            f"pamica_s={pamica_seconds:.1f} amica_s={amica_seconds:.1f}"
        )
    _assert_e1(minimum, median, amari)
