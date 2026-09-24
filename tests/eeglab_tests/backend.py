"""Explicit test backends; MATLAB transport is independent of EEG file I/O."""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import warnings

import numpy as np
from numpy.exceptions import ComplexWarning
from scipy.io import loadmat, savemat


def call_python(name, *args, nargout=1, **kwargs):
    """Resolve lazily, treating Python tuples as MATLAB's multiple outputs."""
    function = getattr(importlib.import_module("eegprep"), name)
    result = function(*args, **kwargs)
    if nargout == 0:
        return None
    if isinstance(result, tuple) and len(result) >= nargout:
        return result[0] if nargout == 1 else result[:nargout]
    if nargout > 1:
        raise AssertionError(f"{name} did not return {nargout} outputs")
    return result


def _decode(value, typed):
    if value.dtype.names and value.size:
        decoded = np.empty(value.shape, dtype=value.dtype)
        for index in np.ndindex(value.shape):
            for field in value.dtype.names:
                decoded[field][index] = _decode(value[field][index], typed[field][index])
        # A scalar MATLAB struct becomes a dict, never a squeezed numeric array.
        if value.shape == (1, 1):
            return {field: decoded[field][0, 0] for field in value.dtype.names}
        return decoded
    if value.dtype == object:
        decoded = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            decoded[index] = _decode(value[index], typed[index])
        return decoded
    if value.dtype.kind in "US" and value.size == 1:
        return value.item()
    return value if np.iscomplexobj(value) else typed


def call_matlab(engine, name, *args, nargout=1, **kwargs):
    """Call MATLAB without production save/load, dtype coercion, or squeezing.

    Use explicitly shaped arrays for MATLAB rows, columns and cells. Struct
    arrays use NumPy structured arrays; a scalar struct is returned as a dict.
    MATLAB name/value options may be positional or supplied as keywords.
    """
    arguments = [*args]
    for key, value in kwargs.items():
        arguments.extend((key, value))
    cells = np.empty((1, len(arguments)), dtype=object)
    for index, value in enumerate(arguments):
        cells[0, index] = value
    with tempfile.TemporaryDirectory(prefix="eegprep_reference_") as directory:
        input_file = Path(directory) / "input.mat"
        output_file = Path(directory) / "output.mat"
        savemat(input_file, {"arguments": cells}, long_field_names=True)
        engine.eegprep_test_call(str(input_file), str(output_file), name, float(nargout), nargout=0)
        if nargout == 0:
            return None
        raw = loadmat(output_file)["outputs"]
        # MATLAB can store integral-valued doubles with an integer MAT storage
        # type. SciPy's MATLAB dtype mode restores their class and logicals,
        # but discards imaginary components. Keep complex values from raw.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ComplexWarning)
            typed = loadmat(output_file, mat_dtype=True)["outputs"]
        outputs = tuple(_decode(value, kind) for value, kind in zip(raw[0], typed[0], strict=True))
        if len(outputs) != nargout:
            raise AssertionError(f"{name} did not return {nargout} outputs")
        return outputs[0] if nargout == 1 else outputs
