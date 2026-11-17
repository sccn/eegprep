"""Export EEG data or ICA activity to a text file."""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.popfunc._file_io import channel_labels
from eegprep.functions.popfunc._pop_utils import format_history_value, is_on, parse_key_value_args


_EXPORT_EXPR_FUNCTIONS = {
    "abs": np.abs,
    "clip": np.clip,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "nan_to_num": np.nan_to_num,
    "sqrt": np.sqrt,
}
_EXPORT_EXPR_KEYWORDS = {
    "clip": {"a_min", "a_max", "max", "min"},
    "nan_to_num": {"nan", "neginf", "posinf"},
}
_MAX_POWER_EXPONENT = 12


def pop_export(
    EEG: dict[str, Any],
    filename: str | Path | None = None,
    *args: Any,
    renderer: Any = None,
    **kwargs: Any,
) -> str:
    """Export EEG data or ICA activity to a delimited text file.

    With no filename, the export dialog from ``pop_export.m`` lines 77-115 is
    shown. Cancel returns an empty command.
    """
    if filename is None:
        selected = _export_dialog_options(renderer)
        if selected is None:
            return ""
        filename, options = selected
    else:
        options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    data = _selected_data(EEG, ica=is_on(options.get("ica", "off")))
    if is_on(options.get("erp", "off")) and data.ndim == 3:
        data = data.mean(axis=2)
    elif data.ndim == 3:
        data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    if options.get("expr"):
        data = _apply_expression(data, str(options["expr"]))
        if data.ndim == 3:
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    use_time = is_on(options.get("time", "on"))
    if use_time:
        time = np.tile(
            np.linspace(float(EEG.get("xmin", 0)), float(EEG.get("xmax", 0)), int(EEG["pnts"]))
            / _time_unit(options.get("timeunit", 1e-3)),
            int(EEG.get("trials", 1)) if not is_on(options.get("erp", "off")) else 1,
        )
        data = np.vstack([time, data])
    separator = str(options.get("separator", "\t"))
    precision = int(options.get("precision", 7))
    use_elec = is_on(options.get("elec", "on"))
    ica = is_on(options.get("ica", "off"))
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    transpose = is_on(options.get("transpose", "off"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=separator)
        if transpose:
            if use_elec:
                writer.writerow(_transpose_header(EEG, data.shape[0], ica=ica, use_time=use_time))
            writer.writerows(_format_row(row, precision) for row in data.T)
        else:
            for index, row in enumerate(data):
                values = _format_row(row, precision)
                if use_elec:
                    values = [_row_label(EEG, index, ica=ica, use_time=use_time), *values]
                writer.writerow(values)
    return _history_command(filename, options)


def pop_export_dialog_spec() -> DialogSpec:
    """Return the export dialog from ``pop_export.m`` lines 77-99."""
    browse = CallbackSpec(
        "select_file",
        {
            "button": "browse",
            "target": "filename",
            "mode": "save",
            "caption": "Select a text file",
            "filter": "All files (*)",
        },
    )
    return DialogSpec(
        title="Export data - pop_export()",
        function_name="pop_export",
        eeglab_source="functions/popfunc/pop_export.m",
        geometry=(
            [1, 2, 0.5],
            *[[1.7, 0.25, 0.7, 0.6]] * 5,
            [3, 0.7, 0.8, 0.6],
            [3, 2, 0.05, 0.05],
            [3, 2, 0.05, 0.05],
        ),
        controls=(
            ControlSpec("text", "Output file name"),
            ControlSpec("edit", tag="filename"),
            ControlSpec("pushbutton", "Browse", tag="browse", callback=browse),
            ControlSpec("text", "Export ICA activities instead of EEG data:"),
            ControlSpec("checkbox", tag="ica", value=False),
            ControlSpec("text", ""),
            ControlSpec("text", ""),
            ControlSpec("text", "Export ERP average instead of trials:"),
            ControlSpec("checkbox", tag="erp", value=False),
            ControlSpec("text", ""),
            ControlSpec("text", ""),
            ControlSpec("text", "Transpose matrix (elec -> cols):"),
            ControlSpec("checkbox", tag="transpose", value=True),
            ControlSpec("text", ""),
            ControlSpec("text", ""),
            ControlSpec("text", "Export channel labels/component numbers:"),
            ControlSpec("checkbox", tag="elec", value=True),
            ControlSpec("text", ""),
            ControlSpec("text", ""),
            ControlSpec("text", "Use comma separator (csv) instead of tab:"),
            ControlSpec("checkbox", tag="separator", value=False),
            ControlSpec("text", ""),
            ControlSpec("text", ""),
            ControlSpec("text", "Export time values:"),
            ControlSpec("checkbox", tag="time", value=True),
            ControlSpec("text", "Unit (re. sec)"),
            ControlSpec("edit", tag="timeunit", value="1E-3"),
            ControlSpec("text", "Number of significant digits to output:"),
            ControlSpec("edit", tag="precision", value="4"),
            ControlSpec("text", ""),
            ControlSpec("text", ""),
            ControlSpec("text", "Apply an expression to the output (see 'expr' help):"),
            ControlSpec("edit", tag="expr"),
            ControlSpec("text", ""),
            ControlSpec("text", ""),
        ),
        known_differences=("The label says significant digits, but pop_export.m:179 writes %.Nf decimal places.",),
    )


def _export_dialog_options(renderer: Any) -> tuple[str, dict[str, Any]] | None:
    """Decode the dialog the way ``pop_export.m`` lines 104-115 do."""
    result = inputgui(pop_export_dialog_spec(), renderer=renderer)
    if not result:
        return None
    filename = str(result.get("filename") or "").strip()
    if not filename:
        raise ValueError("pop_export: File name required. Enter an output file name, then press Ok.")
    options: dict[str, Any] = {}
    if result.get("ica"):
        options["ica"] = "on"
    if result.get("erp"):
        options["erp"] = "on"
    if result.get("transpose"):
        options["transpose"] = "on"
    if not result.get("elec", True):
        options["elec"] = "off"
    if result.get("separator"):
        options["separator"] = ","
    if not result.get("time", True):
        options["time"] = "off"
    timeunit = str(result.get("timeunit") or "1E-3").strip()
    if timeunit.casefold() != "1e-3":
        options["timeunit"] = _time_unit(timeunit)
    precision = str(result.get("precision") or "7").strip()
    if precision != "7":
        options["precision"] = _precision(precision)
    expression = str(result.get("expr") or "").strip()
    if expression:
        options["expr"] = expression
    return filename, options


def _time_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pop_export: time unit {value!r} is not a number. Enter a value such as 1E-3.") from exc
    if number <= 0:
        raise ValueError(f"pop_export: time unit {value!r} must be greater than 0. Enter a value such as 1E-3.")
    return number


def _precision(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"pop_export: precision {value!r} is not an integer. Enter the number of decimal places, such as 7."
        ) from exc
    if number < 0:
        raise ValueError(f"pop_export: precision {value!r} must be 0 or greater. Enter a value such as 7.")
    return number


def _names(EEG: dict[str, Any], count: int, *, ica: bool) -> list[str]:
    chanlocs = EEG.get("chanlocs")
    if isinstance(chanlocs, np.ndarray):
        chanlocs = [] if chanlocs.size == 0 else chanlocs.tolist()
    if ica or not chanlocs:
        return [str(index) for index in range(1, count + 1)]
    labels = channel_labels(EEG)
    return [labels[index] if index < len(labels) else str(index + 1) for index in range(count)]


def _transpose_header(EEG: dict[str, Any], n_rows: int, *, ica: bool, use_time: bool) -> list[str]:
    count = int(EEG.get("nbchan") or max(0, n_rows - (1 if use_time else 0)))
    names = _names(EEG, count, ica=ica)
    return ["Time", *names] if use_time else names


def _row_label(EEG: dict[str, Any], index: int, *, ica: bool, use_time: bool) -> str:
    # pop_export.m:217 writes a leading space on the non-transposed time label.
    if use_time and index == 0:
        return " Time"
    label_index = index - 1 if use_time else index
    return _names(EEG, label_index + 1, ica=ica)[label_index]


def _selected_data(EEG: dict[str, Any], *, ica: bool) -> np.ndarray:
    if not ica:
        return np.asarray(EEG["data"])
    icaact = np.asarray(EEG.get("icaact", []))
    if icaact.size:
        return icaact
    weights = np.asarray(EEG.get("icaweights", []))
    sphere = np.asarray(EEG.get("icasphere", []))
    if not weights.size or not sphere.size:
        raise ValueError("No ICA activity or ICA weights are available")
    channels = np.asarray(EEG.get("icachansind", np.arange(EEG["nbchan"])), dtype=int)
    data = np.asarray(EEG["data"])[channels, ...]
    unmixing = finite_matmul(weights, sphere)
    activations = finite_matmul(unmixing, data.reshape((len(channels), -1)))
    return activations.reshape((activations.shape[0], int(EEG["pnts"]), int(EEG.get("trials", 1))))


def _format_row(row: np.ndarray, precision: int) -> list[str]:
    return [f"{float(value):.{precision}f}" for value in row]


def _apply_expression(data: np.ndarray, expression: str) -> np.ndarray:
    text = expression.strip().rstrip(";")
    if not text:
        return data
    expression_node = _parse_expression(text)
    _validate_expression(expression_node)
    functions = SimpleNamespace(**_EXPORT_EXPR_FUNCTIONS)
    namespace = {"x": data, "np": functions, "numpy": functions, **_EXPORT_EXPR_FUNCTIONS}
    try:
        compiled = compile(ast.fix_missing_locations(ast.Expression(expression_node)), "<pop_export expr>", "eval")
        evaluated = eval(compiled, {"__builtins__": {}}, namespace)
    except Exception as exc:
        raise ValueError(f"pop_export expr could not be evaluated: {exc}") from exc
    output = np.asarray(evaluated)
    if output.ndim not in {2, 3}:
        raise ValueError("pop_export expr must leave x as a 2-D or 3-D array")
    return output


def _parse_expression(source: str) -> ast.expr:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"pop_export expr is not a valid Python expression: {exc.msg}") from exc
    if len(tree.body) != 1:
        raise ValueError("pop_export expr must contain one expression or x assignment")
    statement = tree.body[0]
    if isinstance(statement, ast.Expr):
        return statement.value
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise ValueError("pop_export expr may only assign to x")
        if statement.targets[0].id != "x":
            raise ValueError("pop_export expr may only assign to x")
        return statement.value
    raise ValueError("pop_export expr must contain one expression or x assignment")


def _validate_expression(tree: ast.expr) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _validate_expression_call(node)
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            _validate_power_exponent(node.right)
            continue
        if isinstance(node, ast.Attribute):
            if _allowed_numpy_attribute(node):
                continue
            raise ValueError(f"pop_export expr contains unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in {"x", "np", "numpy", *_EXPORT_EXPR_FUNCTIONS}:
                raise ValueError(f"pop_export expr references unknown name: {node.id}")
            continue
        if isinstance(
            node,
            (
                ast.Load,
                ast.Constant,
                ast.UnaryOp,
                ast.BinOp,
                ast.BoolOp,
                ast.Compare,
                ast.IfExp,
                ast.Subscript,
                ast.Slice,
                ast.Tuple,
                ast.List,
                ast.keyword,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.Pow,
                ast.MatMult,
                ast.UAdd,
                ast.USub,
                ast.Not,
                ast.And,
                ast.Or,
                ast.Eq,
                ast.NotEq,
                ast.Lt,
                ast.LtE,
                ast.Gt,
                ast.GtE,
            ),
        ):
            continue
        raise ValueError(f"pop_export expr contains unsupported syntax: {type(node).__name__}")


def _validate_expression_call(node: ast.Call) -> None:
    name = _expression_call_name(node.func)
    if name in _EXPORT_EXPR_FUNCTIONS:
        _validate_expression_keywords(name, node.keywords)
        return
    raise ValueError("pop_export expr only supports arithmetic and selected NumPy numeric functions")


def _expression_call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and _allowed_numpy_attribute(node):
        return node.attr
    return None


def _validate_expression_keywords(function_name: str, keywords: list[ast.keyword]) -> None:
    allowed = _EXPORT_EXPR_KEYWORDS.get(function_name, set())
    for keyword in keywords:
        if keyword.arg is None:
            raise ValueError("pop_export expr does not support ** keyword expansion")
        if keyword.arg not in allowed:
            allowed_text = ", ".join(sorted(allowed)) or "none"
            raise ValueError(
                f"pop_export expr does not support keyword argument {keyword.arg!r} "
                f"for {function_name}; allowed keywords: {allowed_text}"
            )


def _validate_power_exponent(node: ast.expr) -> None:
    exponent = _literal_number(node)
    if exponent is None:
        raise ValueError("pop_export expr power exponents must be numeric constants")
    if abs(exponent) > _MAX_POWER_EXPONENT:
        raise ValueError(
            f"pop_export expr power exponents must be between -{_MAX_POWER_EXPONENT} and {_MAX_POWER_EXPONENT}"
        )


def _literal_number(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal_number(node.operand)
        if value is None:
            return None
        return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _allowed_numpy_attribute(node: ast.Attribute) -> bool:
    return isinstance(node.value, ast.Name) and node.value.id in {"np", "numpy"} and node.attr in _EXPORT_EXPR_FUNCTIONS


def _history_command(filename: str | Path, options: dict[str, Any]) -> str:
    pieces = [format_history_value(str(filename))]
    for key in ["ica", "time", "timeunit", "elec", "transpose", "erp", "expr", "precision", "separator"]:
        if key in options:
            pieces.extend([format_history_value(key), format_history_value(options[key])])
    return f"LASTCOM = pop_export(EEG, {', '.join(pieces)});"
