"""Replay the shrunk 20261017 ``pop_resample|nan-mask|na`` input."""

from __future__ import annotations

import copy

import numpy as np

from eegprep.functions.adminfunc.eeglabcompat import MatlabWrapper
from tests.fuzz.harness import (
    EEGSpec,
    FuzzCase,
    compare_data,
    data_sha256,
    dispatch_matlab,
    dispatch_python,
    freeze_params,
    load_tolerances,
    materialize_eeg,
    resume_eeg_options,
    suspend_option_single,
    tolerance_for,
)

_RESAMPLE_NAN_20261017 = FuzzCase(
    "pop_resample",
    EEGSpec(
        channels=tuple(range(25)),
        start=405,
        length=598,
        decimate=1,
        nan_channel=0,
        nan_start=215,
        nan_length=1,
    ),
    freeze_params({"freq": 100}),
)
_INPUT_SHA256 = "5ce0a96177e06737bc8b167dbfdbfd273349495016d2003ccbed3e70880537a2"


def _nonfinite_indices(data: np.ndarray) -> list[int]:
    return [int(index) for index in np.flatnonzero(~np.isfinite(data))]


def test_20261017_resample_nan_mask_input_matches_the_shrunk_artifact():
    eeg = materialize_eeg(_RESAMPLE_NAN_20261017)
    assert data_sha256(np.asarray(eeg["data"])) == _INPUT_SHA256


def test_replay_20261017_resample_nan_mask(eeglab_matlab_engine):
    """Replay issue #512. Run with --eeglab-backend=matlab -s."""
    engine = MatlabWrapper(eeglab_matlab_engine)
    snapshot = suspend_option_single(eeglab_matlab_engine)
    tolerances = load_tolerances()
    try:
        case = _RESAMPLE_NAN_20261017
        eeg = materialize_eeg(case)
        assert data_sha256(np.asarray(eeg["data"])) == _INPUT_SHA256
        python = np.asarray(dispatch_python(case, copy.deepcopy(eeg))["data"], dtype=np.float64)
        oracle = np.asarray(dispatch_matlab(engine, case, copy.deepcopy(eeg))["data"], dtype=np.float64)
        diverged, error_class, max_abs, max_rel, detail = compare_data(
            python,
            oracle,
            float(eeg["srate"]),
            float(eeg["srate"]),
            tolerance_for(case.function, tolerances),
        )
        python_bad = _nonfinite_indices(python[0])
        oracle_bad = _nonfinite_indices(oracle[0])
        print(
            f"pop_resample diverged={diverged} class={error_class} abs={max_abs} rel={max_rel} "
            f"python_nonfinite={python_bad[:8]}..{python_bad[-8:]} count={len(python_bad)} "
            f"oracle_nonfinite={oracle_bad[:8]}..{oracle_bad[-8:]} count={len(oracle_bad)} {detail}"
        )
        print(f"python_indices={python_bad}")
        print(f"oracle_indices={oracle_bad}")
        assert diverged is False, detail
    finally:
        resume_eeg_options(eeglab_matlab_engine, snapshot)
