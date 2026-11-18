"""Print standardBESA position distances and leadfield error against a MATLAB reference.

The MATLAB side is ``tools/dipfit_standard_besa_parity.m``. It writes a MAT file;
this script loads that file, refits components 1-3, and reprints the comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.dipfit.forward.ft_compute_leadfield import ft_compute_leadfield
from eegprep.plugins.dipfit.forward.ft_prepare_vol_sens import ft_prepare_vol_sens
from eegprep.plugins.dipfit.forward.standard_besa import standard_besa_headmodel
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_multifit import pop_multifit


def _array(mapping: dict, name: str) -> np.ndarray:
    value = np.asarray(mapping[name], dtype=float)
    return np.squeeze(value)


def main(mat_path: str) -> None:
    loaded = loadmat(mat_path, squeeze_me=True)
    reference = {key: value for key, value in loaded.items() if not key.startswith("__")}
    raw = _array(reference, "elec_raw")
    projected = _array(reference, "elec_projected")
    dipoles = _array(reference, "dippos")
    matlab_leadfield = _array(reference, "leadfield")
    matlab_positions = _array(reference, "fitpos").reshape(3, 3)
    matlab_rv = _array(reference, "fitrv").reshape(3)
    if raw.shape[0] == 3 and raw.shape[1] != 3:
        raw = raw.T
        projected = projected.T
    if dipoles.shape[0] == 3 and dipoles.shape[1] != 3:
        dipoles = dipoles.T

    dataset = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"
    fitted = pop_multifit(
        pop_dipfit_settings(pop_loadset(dataset), model="standardBESA"),
        [1, 2, 3],
        "threshold",
        100,
        "dipplot",
        "off",
    )
    positions = np.vstack([np.asarray(fitted["dipfit"]["model"][index]["posxyz"], dtype=float) for index in range(3)])
    python_rv = np.array([float(fitted["dipfit"]["model"][index]["rv"]) for index in range(3)])
    _model, sensors = ft_prepare_vol_sens(standard_besa_headmodel(), {"elecpos": raw})
    python_projected = np.asarray(sensors["elecpos"], dtype=float)
    python_leadfield = ft_compute_leadfield(dipoles, {"elecpos": raw}, standard_besa_headmodel())
    leadfield_scale = float(np.max(np.abs(matlab_leadfield)))
    leadfield_error = float(np.max(np.abs(python_leadfield - matlab_leadfield)) / leadfield_scale)
    projection_error = float(np.max(np.abs(python_projected - projected)))

    print("component  distance_mm  eegprep_rv  eeglab_rv  eegprep_xyz_mm  eeglab_xyz_mm")
    for index in range(3):
        distance = float(np.linalg.norm(positions[index] - matlab_positions[index]))
        print(
            f"{index + 1:9d}  {distance:11.4f}  {python_rv[index]:10.6f}  {matlab_rv[index]:9.6f}  "
            f"{np.array2string(positions[index], precision=4)}  "
            f"{np.array2string(matlab_positions[index], precision=4)}"
        )
    print(f"leadfield_max_relative_error {leadfield_error:.6e}")
    print(f"electrode_projection_max_abs_mm {projection_error:.6e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: dipfit_standard_besa_parity.py REFERENCE.mat")
    main(sys.argv[1])
