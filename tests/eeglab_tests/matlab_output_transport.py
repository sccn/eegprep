"""MATLAB shapes for values that leave through the Python test backend.

Library functions return 1-D vectors. This module adds the row, column, or
1-by-1 shape only for ``call_python`` comparisons.
"""

from __future__ import annotations

from dataclasses import replace
from numbers import Real
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.matlab_outputs import as_column, as_row, empty_matrix


def adapt_matlab_output(name: str, result: Any) -> Any:
    adapter = _ADAPTERS.get(name)
    if adapter is None:
        return result
    return adapter(result)


def _rows(result: Any) -> Any:
    if isinstance(result, tuple):
        return tuple(as_row(item) for item in result)
    return as_row(result)


def _row_if_vector(result: Any) -> Any:
    array = np.asarray(result)
    if array.dtype == object or isinstance(result, str):
        return result
    if array.ndim < 2:
        return as_row(array)
    return result


def _numeric_scalar(value: Any) -> Any:
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, Real):
        return np.array([[float(value)]])
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and value.ndim < 2 and value.size <= 1:
        return np.asarray(value, dtype=float).reshape(1, 1) if value.size else value.reshape(1, 0)
    return value


def _index_row(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return empty_matrix()
    return as_row(array.astype(float, copy=False))


def _cell_row(items: list[Any]) -> np.ndarray:
    if not items:
        return np.empty((0, 0), dtype=object)
    row = np.empty((1, len(items)), dtype=object)
    row[0, :] = items
    return row


def _adapt_chancoresp(result: Any) -> Any:
    if not isinstance(result, tuple):
        return result
    adapted = []
    for item in result:
        if isinstance(item, str):
            adapted.append(item)
        elif isinstance(item, list) and item and isinstance(item[0], str):
            adapted.append(_cell_row(item))
        elif isinstance(item, list):
            adapted.append(_index_row(item))
        else:
            adapted.append(item)
    return tuple(adapted)


def _adapt_getepochevent(result: Any) -> Any:
    values, grouped = result
    try:
        numeric: Any = as_row(np.asarray(values, dtype=float))
    except (TypeError, ValueError):
        numeric = values
    grouped_list = list(grouped)
    cells = np.empty((1, len(grouped_list)), dtype=object)
    for index, item in enumerate(grouped_list):
        array = np.asarray(item)
        cells[0, index] = array.reshape(1, -1) if array.ndim < 2 else array
    return numeric, cells


def _adapt_finputcheck(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    for key, value in list(result.items()):
        result[key] = _numeric_scalar(value)
    return result


def _adapt_readbdf(result: Any) -> Any:
    if not isinstance(result, tuple) or not result or not isinstance(result[0], dict):
        return result
    data = dict(result[0])
    if "Idx" in data:
        data["Idx"] = as_row(data["Idx"])
    return (data, *result[1:])


def _adapt_rejsuperpose(result: Any) -> Any:
    paired = isinstance(result, tuple)
    dataset = result[0] if paired else result
    if not isinstance(dataset, dict) or "reject" not in dataset:
        return result
    copied = dict(dataset)
    reject = dict(dataset["reject"])
    if "rejglobal" in reject:
        reject["rejglobal"] = as_row(reject["rejglobal"])
    copied["reject"] = reject
    if paired:
        return (copied, *result[1:])
    return copied


def _adapt_loadtxt(result: Any) -> Any:
    array = np.asarray(result)
    if array.ndim == 1 and array.size:
        return as_row(array)
    return result


def _adapt_mattocell(result: Any) -> Any:
    if not isinstance(result, list):
        return result
    values = np.asarray(result)
    if values.size == 0:
        return np.empty(values.shape if values.ndim else (0, 0), dtype=object)
    cells = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        cells[index] = np.array([[values[index]]])
    return cells


def _adapt_parsetxt(result: Any) -> Any:
    if not isinstance(result, list):
        return result
    return _cell_row(result)


def _adapt_quantile(result: Any) -> Any:
    array = np.asarray(result)
    if array.ndim == 0:
        return as_row(array)
    return result


def _adapt_kmeans(result: Any) -> Any:
    labels, centers, previous, matrix = result
    return as_column(labels), centers, as_column(previous), matrix


def _adapt_uniquef(result: Any) -> Any:
    values, counts, indices = result
    return as_column(values), as_column(counts), as_column(indices)


def _adapt_pcsquash(result: Any) -> Any:
    vectors, values, compressed, mean = result
    return vectors, as_row(values), compressed, as_row(mean)


def _adapt_eegrej(result: Any) -> Any:
    data, duration, events, bounds = result
    return data, as_row(duration), events, as_column(bounds)


def _adapt_eegthresh(result: Any) -> Any:
    accepted, rejected, newsignal, elec = result
    return as_row(accepted), as_row(rejected), newsignal, elec


def _adapt_movav(result: Any) -> Any:
    output, output_x = result
    return output, as_row(output_x)


def _adapt_chancenter(result: Any) -> Any:
    x_arr, y_arr, z_arr, center, optimize = result
    return as_row(x_arr), as_row(y_arr), as_row(z_arr), as_row(center), optimize


def _adapt_realproba(result: Any) -> Any:
    probabilities, distribution = result
    return probabilities, as_row(distribution)


def _adapt_concatdata(result: Any) -> Any:
    return replace(result, lengths=as_row(result.lengths))


def _adapt_pop_chancenter(result: Any) -> Any:
    center = getattr(result, "center", None)
    if center is not None:
        result.center = as_row(center)
    return result


def _adapt_matchchans(result: Any) -> Any:
    selected, distances, locations = result
    return as_column(selected), as_column(distances), locations


def _adapt_eegplot2trial(result: Any) -> Any:
    trialrej, elecrej = result
    return as_row(trialrej), elecrej


def _adapt_getchanlist(result: Any) -> Any:
    if isinstance(result, list):
        return _index_row(result)
    return result


_ADAPTERS = {
    "angtimewarp": _rows,
    "cart2topo": _rows,
    "chancenter": _adapt_chancenter,
    "concatdata": _adapt_concatdata,
    "covary": _rows,
    "datlim": _rows,
    "eeg_getepochevent": _adapt_getepochevent,
    "eeg_matchchans": _adapt_matchchans,
    "eeg_ms2f": _row_if_vector,
    "eeg_rejsuperpose": _adapt_rejsuperpose,
    "eeg_urlatency": _row_if_vector,
    "eegplot2trial": _adapt_eegplot2trial,
    "eegrej": _adapt_eegrej,
    "eegthresh": _adapt_eegthresh,
    "entropy_rej": _rows,
    "finputcheck": _adapt_finputcheck,
    "getchanlist": _adapt_getchanlist,
    "getkeyval": _row_if_vector,
    "hungarian": _rows,
    "kmeanscluster": _adapt_kmeans,
    "kurt": _rows,
    "loadtxt": _adapt_loadtxt,
    "mattocell": _adapt_mattocell,
    "movav": _adapt_movav,
    "nan_mean": _rows,
    "nan_std": _rows,
    "numdim": _row_if_vector,
    "parsetxt": _adapt_parsetxt,
    "pcsquash": _adapt_pcsquash,
    "pop_chancenter": _adapt_pop_chancenter,
    "pop_chancoresp": _adapt_chancoresp,
    "quantile": _adapt_quantile,
    "readbdf": _adapt_readbdf,
    "realproba": _adapt_realproba,
    "uniquef": _adapt_uniquef,
}
