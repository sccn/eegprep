"""Template-file coregistration and the EEGLAB traditional warp."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc.coregister import coregister, read_electrode_file
from eegprep.plugins.dipfit._coordinates import warp_apply, warp_optim
from tests.eeglab_tests.backend import call_matlab


_ELC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "eegprep"
    / "resources"
    / "dipfit"
    / "standard_BEM"
    / "elec"
    / "standard_1005.elc"
)
_SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"
# Same allowance as test_coregister_traditional_warp_matches_eeglab.
_MATLAB_ATOL = 1e-3
_MATLAB_RTOL = 1e-4

_ELP = """%F 0 90 0
%F -90 0 0
%F 90 0 0
%N Cz
0 0 85
%N Fp1
0 80 10
%N Oz
0 -80 10
"""


def test_coregister_reads_elc_and_elp_templates(tmp_path):
    elc = read_electrode_file(_ELC)
    assert len(elc.labels) > 300
    assert any(label.lower() == "cz" for label in elc.labels)

    elp_path = tmp_path / "standard-10-5-cap385.elp"
    elp_path.write_text(_ELP, encoding="utf-8")
    elp = read_electrode_file(elp_path)
    assert [label.lower() for label in elp.labels[:3]] == ["nz", "lpa", "rpa"]
    assert any(label.lower() == "cz" for label in elp.labels)

    dataset = pop_loadset(str(_SAMPLE))
    fitted = coregister(dataset["chanlocs"], str(_ELC), warp="auto")
    assert fitted.transform.shape == (9,)
    assert np.isfinite(fitted.transform).all()


@pytest.mark.matlab
@pytest.mark.timeout(1800)
def test_coregister_traditional_warp_matches_eeglab(eeglab_matlab_engine, request):
    eeglab_root = Path(request.config.getoption("--eeglab-root"))
    templates = [
        _ELC,
        eeglab_root / "plugins" / "dipfit" / "standard_BESA" / "standard-10-5-cap385.elp",
    ]
    dataset = pop_loadset(str(_SAMPLE))
    for template in templates:
        if not template.is_file():
            raise AssertionError(f"coregister template is missing: {template}")
        reference = call_matlab(
            eeglab_matlab_engine,
            "eegprep_coregister_parity",
            str(_SAMPLE),
            str(template),
            nargout=1,
        )
        fitted = coregister(dataset["chanlocs"], str(template), warp="auto")
        matlab = np.asarray(reference["transform"], dtype=float).reshape(-1)
        np.testing.assert_allclose(fitted.transform, matlab, atol=_MATLAB_ATOL, rtol=_MATLAB_RTOL)


_SYNTHETIC_SOURCE = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, -0.2],
        [1.0, 0.0, -0.2],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


@pytest.mark.matlab
@pytest.mark.timeout(600)
def test_synthetic_traditional_warp_matches_fieldtrip(eeglab_matlab_engine, request):
    eeglab_root = Path(request.config.getoption("--eeglab-root"))
    reference = call_matlab(
        eeglab_matlab_engine,
        "eegprep_warp_optim_parity",
        str(eeglab_root),
        nargout=1,
    )
    degrees = np.asarray([5.0, -3.0, 2.0, *np.rad2deg([0.05, -0.04, 0.03]), 2.0, 1.5, 1.2])
    target = warp_apply(degrees, _SYNTHETIC_SOURCE, "traditional")
    recovered, params = warp_optim(_SYNTHETIC_SOURCE, target, "traditional")
    matlab_params = np.asarray(reference["params"], dtype=float).reshape(-1)
    matlab_points = np.asarray(reference["points"], dtype=float).reshape(-1, 3)
    np.testing.assert_allclose(params, matlab_params, atol=_MATLAB_ATOL, rtol=_MATLAB_RTOL)
    np.testing.assert_allclose(recovered, matlab_points, atol=_MATLAB_ATOL, rtol=_MATLAB_RTOL)


@pytest.mark.matlab
@pytest.mark.timeout(600)
def test_empty_icawinv_matches_eeglab(eeglab_matlab_engine):
    weights = np.array([[0.25, -0.5], [0.5, 0.25]], dtype=float)
    sphere = np.array([[0.8, 0.1], [-0.2, 1.1]], dtype=float)
    reference = call_matlab(
        eeglab_matlab_engine,
        "eegprep_checkset_empty_icawinv",
        weights,
        sphere,
        nargout=1,
    )
    eeg = {
        "data": np.zeros((2, 8)),
        "nbchan": 2,
        "pnts": 8,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 7 / 100,
        "icaweights": weights,
        "icasphere": sphere,
        "icawinv": np.array([]),
        "icaact": np.array([]),
        "icachansind": np.array([0, 1]),
    }
    checked = eeg_checkset(eeg)
    np.testing.assert_allclose(checked["icawinv"], np.asarray(reference["icawinv"], dtype=float), atol=1e-12, rtol=0)
