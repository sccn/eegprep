"""Explicit test backends; MATLAB transport is independent of EEG file I/O."""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import warnings

import numpy as np
from numpy.exceptions import ComplexWarning
from scipy.io import loadmat, savemat
from scipy.sparse import issparse


_MAT_INLINE_LIMIT = 2**30
_MAT_PART_LIMIT = 2**28


def _input_nbytes(value):
    # Include conservative container/header overhead, not just numerical leaves.
    if isinstance(value, dict):
        return 128 + sum(4 * len(key) + _input_nbytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return 128 + sum(_input_nbytes(item) for item in value)
    if isinstance(value, (np.ndarray, np.void)):
        if value.dtype.names:
            return 128 + sum(_input_nbytes(value[field]) for field in value.dtype.names)
        if value.dtype == object:
            return 128 + sum(_input_nbytes(item) for item in value.flat)
        return 128 + value.nbytes
    if issparse(value):
        return 128 + value.nnz * (value.dtype.itemsize + 16) + 8 * (max(value.shape) + 1)
    return 128 + (4 * len(value) if isinstance(value, str) else 0)


def _spill_input_arrays(value, directory, leaf_limit_bytes, part_limit_bytes):
    if isinstance(value, dict):
        if tuple(value) == ("eegprep_test_array_mat_v1", "shape"):
            return value
        return {
            key: _spill_input_arrays(item, directory, leaf_limit_bytes, part_limit_bytes) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, np.void)):
        try:
            value = np.asarray(value)
        except ValueError:
            value = np.asarray(value, dtype=object)
    if not isinstance(value, np.ndarray):
        return value
    if value.dtype.names:
        result = np.empty(value.shape, dtype=[(field, object) for field in value.dtype.names])
        for index in np.ndindex(value.shape):
            for field in value.dtype.names:
                result[field][index] = _spill_input_arrays(
                    value[field][index], directory, leaf_limit_bytes, part_limit_bytes
                )
        return result
    if value.dtype == object:
        result = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            result[index] = _spill_input_arrays(value[index], directory, leaf_limit_bytes, part_limit_bytes)
        return result
    if value.dtype.kind not in "buifc" or value.nbytes <= leaf_limit_bytes:
        return value
    paths = []
    # Buffered F-order iteration avoids making a second full copy of a large
    # C-ordered or strided input just to flatten it for MATLAB linear indexing.
    for part in np.nditer(
        value,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="F",
        buffersize=max(1, part_limit_bytes // value.itemsize),
    ):
        with tempfile.NamedTemporaryFile(dir=directory, prefix="input-array-", suffix=".mat", delete=False) as file:
            savemat(file.file, {"part": part})
            paths.append(Path(file.name).name)
    shape = (1, value.size) if value.ndim < 2 else value.shape
    return {"eegprep_test_array_mat_v1": np.array([paths], dtype=object), "shape": np.array([shape], dtype=float)}


def _save_matlab_arguments(filename, arguments, inline_limit_bytes=_MAT_INLINE_LIMIT, part_limit_bytes=_MAT_PART_LIMIT):
    if _input_nbytes(arguments) > inline_limit_bytes:
        arguments = _spill_input_arrays(arguments, filename.parent, min(2**20, inline_limit_bytes), part_limit_bytes)
        if _input_nbytes(arguments) > inline_limit_bytes:
            arguments = _spill_input_arrays(arguments, filename.parent, 1, part_limit_bytes)
        if _input_nbytes(arguments) >= 2**31:
            raise ValueError("Input metadata still exceeds the MAT variable limit after array sharding")
    savemat(filename, {"arguments": arguments}, long_field_names=True)


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
        _save_matlab_arguments(input_file, cells)
        engine.eegprep_test_call(str(input_file), str(output_file), name, float(nargout), nargout=0)
        if nargout == 0:
            return None
        outputs = tuple(_load_matlab_variable(output_file, "outputs")[0])
        if len(outputs) != nargout:
            raise AssertionError(f"{name} did not return {nargout} outputs")
        return outputs[0] if nargout == 1 else outputs
