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


def _decode(value, typed, directory=None):
    if value.dtype.names and value.size:
        if value.dtype.names == ("eegprep_test_array_mat_v1", "shape") and value.shape == (1, 1):
            paths = _decode(value["eegprep_test_array_mat_v1"][0, 0], typed["eegprep_test_array_mat_v1"][0, 0])
            shape = tuple(value["shape"][0, 0].astype(int).ravel())
            result, offset = None, 0
            for path in paths.flat:
                part = _load_matlab_variable(Path(directory) / path, "part")
                if result is None:
                    result = np.empty(shape, dtype=part.dtype, order="F")
                result.ravel(order="F")[offset : offset + part.size] = part.ravel(order="F")
                offset += part.size
            if result is None or offset != result.size:
                raise AssertionError("MATLAB array sidecars do not match the declared shape")
            return result
        decoded = np.empty(value.shape, dtype=value.dtype)
        for index in np.ndindex(value.shape):
            for field in value.dtype.names:
                decoded[field][index] = _decode(value[field][index], typed[field][index], directory)
        # A scalar MATLAB struct becomes a dict, never a squeezed numeric array.
        if value.shape == (1, 1):
            return {field: decoded[field][0, 0] for field in value.dtype.names}
        return decoded
    if value.dtype == object:
        decoded = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            decoded[index] = _decode(value[index], typed[index], directory)
        return decoded
    if value.dtype.kind in "US" and value.size == 1:
        return value.item()
    return value if np.iscomplexobj(value) else typed


def _load_matlab_variable(filename, variable):
    raw = loadmat(filename)[variable]
    # MATLAB's compact MAT storage may differ from its declared numeric class.
    # The typed read restores classes but drops complex imaginary components.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        typed = loadmat(filename, mat_dtype=True)[variable]
    return _decode(raw, typed, Path(filename).parent)


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
        outputs = tuple(_load_matlab_variable(output_file, "outputs")[0])
        if len(outputs) != nargout:
            raise AssertionError(f"{name} did not return {nargout} outputs")
        return outputs[0] if nargout == 1 else outputs
