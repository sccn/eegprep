"""MATLAB shapes for values that leave through the Python test backend.

Library functions return 1-D vectors. This module adds the row, column, or
1-by-1 shape only for ``call_python`` comparisons.
"""

from __future__ import annotations

from dataclasses import replace
from numbers import Real
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.matlab_inputs import is_cell, struct_records
from eegprep.functions.miscfunc.matlab_outputs import as_column, as_row, empty_matrix


def adapt_matlab_output(
    name: str,
    result: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    nargout: int = 1,
) -> Any:
    if name == "pop_comments":
        return _adapt_pop_comments(result, args, kwargs or {})
    if name == "readelp" and nargout >= 5:
        return _expand_readelp(result)
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
    return as_column(selected), as_column(distances), _location_struct(locations)


def _location_struct(records: list[dict[str, Any]]) -> np.ndarray:
    names = list(records[0])
    row = np.empty((1, len(records)), dtype=[(name, object) for name in names])
    for column, record in enumerate(records):
        row[0, column] = tuple(_location_field(record.get(name)) for name in names)
    return row


def _location_field(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return np.array([[float(value)]])
    if value is None or value == []:
        return np.empty((0, 0))
    return value


def _adapt_pop_comments(result: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    paired = isinstance(result, tuple)
    output = result[0] if paired else result
    if not isinstance(output, str):
        return result
    comments = kwargs.get("EEG", args[0] if args else None)
    newcomments = kwargs.get("newcomments", args[2] if len(args) > 2 else None)
    concat = kwargs.get("concat", args[3] if len(args) > 3 else False)
    if not (is_cell(comments) or is_cell(newcomments)):
        return result
    source = newcomments if newcomments is not None else comments
    lines = _comment_lines(comments) + _comment_lines(source) if _truthy_concat(concat) else _comment_lines(source)
    matrix = _character_matrix(lines)
    if paired:
        return (matrix, *result[1:])
    return matrix


def _truthy_concat(value: Any) -> bool:
    if isinstance(value, str):
        return False
    return bool(value)


def _comment_lines(value: Any) -> list[str]:
    if is_cell(value):
        return ["" if item is None else str(item) for item in np.asarray(value).reshape(-1)]
    if isinstance(value, str):
        text = value.rstrip(" \t")
    elif value is None:
        text = ""
    else:
        text = str(value).rstrip()
    return text.split("\n") if text else []


def _character_matrix(lines: list[str]) -> np.ndarray:
    if not lines:
        return np.empty((0, 0), dtype="U1")
    width = max(len(line) for line in lines)
    return np.array([line.ljust(width) for line in lines])


def _adapt_eegplot2trial(result: Any) -> Any:
    trialrej, elecrej = result
    return as_row(trialrej), elecrej


def _adapt_getchanlist(result: Any) -> Any:
    if isinstance(result, list):
        return _index_row(result)
    return result


_RECORD_FIELDS = ("event", "urevent", "chanlocs", "urchanlocs", "epoch")
_SCALAR_FIELDS = ("nbchan", "trials", "pnts", "srate", "xmin", "xmax")


def _present_records(value: Any) -> Any:
    records = _record_list(value)
    if records is None:
        return None
    if not records:
        return np.empty((0, 0)) if isinstance(value, np.ndarray) else value
    if len(records) == 1:
        return {key: _matlab_field(item) for key, item in records[0].items()}
    return _struct_row(records)


def _record_list(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        return None
    if isinstance(value, list):
        if not value or all(isinstance(item, dict) for item in value):
            return value
        return None
    records = struct_records(value)
    if records is None or isinstance(value, dict):
        return None
    return records


def _struct_row(records: list[dict[str, Any]]) -> np.ndarray:
    names: list[str] = []
    for record in records:
        for name in record:
            if name not in names:
                names.append(name)
    if not names:
        return np.empty((1, len(records)), dtype=object)
    row = np.empty((1, len(records)), dtype=[(name, object) for name in names])
    for column, record in enumerate(records):
        row[0, column] = tuple(_matlab_field(record.get(name)) for name in names)
    return row


def _matlab_field(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype == object or value.dtype.names:
            nested = _present_records(value)
            if nested is not None:
                return nested
        if value.ndim < 2 and np.issubdtype(value.dtype, np.number) and value.size:
            return as_row(value)
        return value
    if isinstance(value, (bool, np.bool_)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.array([[float(value)]])
    if isinstance(value, list):
        nested = _present_records(value)
        if nested is not None:
            return nested
        if value and all(isinstance(item, str) for item in value):
            return _description_column(value)
    if value is None or value == []:
        return np.empty((0, 0))
    return value


def _description_column(lines: list[str]) -> np.ndarray:
    column = np.empty((len(lines), 1), dtype=object)
    column[:, 0] = lines
    return column


def _present_descriptions(value: Any) -> Any:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return _description_column(value)
    return value


def _present_eeg(eeg: dict[str, Any], *, scalars: bool) -> dict[str, Any]:
    output = dict(eeg)
    for field in _RECORD_FIELDS:
        if field not in output:
            continue
        presented = _present_records(output[field])
        if presented is not None:
            output[field] = presented
    if "eventdescription" in output:
        output["eventdescription"] = _present_descriptions(output["eventdescription"])
    if scalars and _has_samples(output):
        for field in _SCALAR_FIELDS:
            if field in output:
                output[field] = _numeric_scalar(output[field])
    return output


def _has_samples(eeg: dict[str, Any]) -> bool:
    data = eeg.get("data")
    if isinstance(data, np.ndarray):
        return data.size > 0
    return data not in (None, [])


def _present_datasets(datasets: list[Any]) -> np.ndarray:
    return _struct_row(
        [_present_eeg(dataset, scalars=True) if isinstance(dataset, dict) else {} for dataset in datasets]
    )


def _present_eeg_value(value: Any, *, scalars: bool) -> Any:
    if isinstance(value, list) and (not value or all(isinstance(item, dict) for item in value)):
        if value and any(key in value[0] for key in ("data", "nbchan", "setname", "trials")):
            return _present_datasets(value)
        presented = _present_records(value)
        return value if presented is None else presented
    if isinstance(value, dict):
        return _present_eeg(value, scalars=scalars)
    return value


def _adapt_eeg_value(result: Any, *, scalars: bool) -> Any:
    if isinstance(result, tuple):
        return (_present_eeg_value(result[0], scalars=scalars), *result[1:])
    return _present_eeg_value(result, scalars=scalars)


def _adapt_loaded_eeg(result: Any) -> Any:
    return _adapt_eeg_value(result, scalars=False)


def _adapt_scalar_eeg(result: Any) -> Any:
    return _adapt_eeg_value(result, scalars=True)


def _adapt_eeg_store(result: Any) -> Any:
    alleeg, eeg, index = result
    return _present_eeg_value(alleeg, scalars=True), _present_eeg_value(eeg, scalars=True), index


def _adapt_eeg_retrieve(result: Any) -> Any:
    eeg, alleeg, index = result
    return (
        _present_eeg_value(eeg, scalars=True),
        _present_eeg_value(alleeg, scalars=True),
        index,
    )


def _adapt_study(result: Any) -> Any:
    if isinstance(result, tuple):
        return (_present_study(result[0]), *result[1:])
    return _present_study(result)


def _present_study(study: Any) -> Any:
    if not isinstance(study, dict) or "design" not in study:
        return study
    output = dict(study)
    presented = _present_records(study["design"])
    if presented is not None:
        output["design"] = _nest_design(presented)
    cluster = _present_records(study.get("cluster"))
    if cluster is not None:
        output["cluster"] = cluster
    return output


def _nest_design(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _nest_design_field(key, item) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.dtype.names:
        nested = np.empty(value.shape, dtype=value.dtype)
        for index in np.ndindex(value.shape):
            nested[index] = tuple(_nest_design_field(name, value[name][index]) for name in value.dtype.names)
        return nested
    return value


def _nest_design_field(name: str, value: Any) -> Any:
    if name in {"variable", "cell", "dataset", "value", "trials"}:
        presented = _present_records(value)
        if presented is not None:
            return _nest_design(presented)
    return value


def _adapt_openbdf(result: Any) -> Any:
    if isinstance(result, tuple):
        return (_column_vectors(result[0]), *result[1:])
    return _column_vectors(result)


def _column_vectors(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _column_vectors(item) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.size and np.issubdtype(value.dtype, np.number):
        return value.reshape(-1, 1)
    return value


def _adapt_readneurodat(result: Any) -> Any:
    if not isinstance(result, tuple):
        return _adapt_records(result)
    struct = _adapt_records(result[0])
    outputs = [struct]
    for item in result[1:]:
        array = np.asarray(item)
        if array.ndim == 1 and array.size:
            outputs.append(array.reshape(1, -1))
        else:
            outputs.append(item)
    return tuple(outputs)


def _adapt_records(result: Any) -> Any:
    if isinstance(result, tuple):
        presented = _present_records(result[0])
        return (result[0] if presented is None else presented, *result[1:])
    presented = _present_records(result)
    return result if presented is None else presented


def _expand_readelp(result: Any) -> Any:
    records = _record_list(result) or []
    struct = _struct_row(records) if records else np.empty((0, 0))
    names = _cell_row([str(record.get("labels", "")) for record in records])

    def coordinate(field: str) -> np.ndarray:
        return as_row(np.array([float(record.get(field, np.nan)) for record in records], dtype=float))

    return struct, names, coordinate("X"), coordinate("Y"), coordinate("Z")


_ADAPTERS = {
    "angtimewarp": _rows,
    "biosig2eeglabevent": _adapt_records,
    "cart2topo": _rows,
    "chancenter": _adapt_chancenter,
    "concatdata": _adapt_concatdata,
    "convertlocs": _adapt_records,
    "covary": _rows,
    "datlim": _rows,
    "eeg_emptyset": _adapt_scalar_eeg,
    "eeg_mergechan": _adapt_records,
    "eeg_mergelocs": _adapt_records,
    "eeg_getepochevent": _adapt_getepochevent,
    "eeg_matchchans": _adapt_matchchans,
    "eeg_ms2f": _row_if_vector,
    "eeg_rejsuperpose": _adapt_rejsuperpose,
    "eeg_retrieve": _adapt_eeg_retrieve,
    "eeg_store": _adapt_eeg_store,
    "eeg_urlatency": _row_if_vector,
    "eegplot2trial": _adapt_eegplot2trial,
    "eegrej": _adapt_eegrej,
    "eegthresh": _adapt_eegthresh,
    "entropy_rej": _rows,
    "finputcheck": _adapt_finputcheck,
    "forcelocs": _adapt_records,
    "getchanlist": _adapt_getchanlist,
    "getkeyval": _row_if_vector,
    "hungarian": _rows,
    "importevent": _adapt_records,
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
    "pop_adjustevents": _adapt_loaded_eeg,
    "pop_chancenter": _adapt_pop_chancenter,
    "pop_chancoresp": _adapt_chancoresp,
    "pop_copyset": _adapt_scalar_eeg,
    "pop_editeventfield": _adapt_loaded_eeg,
    "eeg_interp": _adapt_loaded_eeg,
    "pop_loadset": _adapt_loaded_eeg,
    "pop_select": _adapt_loaded_eeg,
    "pop_newset": _adapt_scalar_eeg,
    "pop_readlocs": _adapt_records,
    "pop_selectevent": _adapt_scalar_eeg,
    "quantile": _adapt_quantile,
    "readbdf": _adapt_readbdf,
    "readeetraklocs": _adapt_records,
    "readegilocs": _adapt_loaded_eeg,
    "openbdf": _adapt_openbdf,
    "readneurodat": _adapt_readneurodat,
    "realproba": _adapt_realproba,
    "std_checkset": _adapt_study,
    "std_makedesign": _adapt_study,
    "std_selectdesign": _adapt_study,
    "uniquef": _adapt_uniquef,
}
