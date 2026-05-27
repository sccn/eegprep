"""EEGLAB-style wrapper for 2-D scalp maps."""

from __future__ import annotations

import math
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import parse_key_value_args, parse_text_tokens
from eegprep.functions.sigprocfunc.topoplot import topoplot


def pop_topoplot(
    EEG: dict[str, Any] | None = None,
    typeplot: int = 1,
    items: Any = None,
    topotitle: str | None = None,
    rowcols: Any = None,
    plotdip: Any = 0,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot EEGLAB-style 2-D scalp maps for ERP latencies or ICA components.

    Args:
        EEG: EEGLAB-style EEG dictionary.
        typeplot: ``1`` for channel ERP latency maps, ``0`` for component maps.
        items: Latencies in ms when ``typeplot=1`` or 1-based component
            indices when ``typeplot=0``. Negative component indices invert map
            polarity and ``NaN`` leaves a blank subplot.
        topotitle: Figure title.
        rowcols: Optional ``[rows, columns]`` layout. Empty uses EEGLAB's
            near-square page geometry.
        plotdip: Accepted for EEGLAB signature compatibility; DIPFIT overlays
            are deferred to the Phase 4 headplot/DIPFIT integration.
        gui: Force or suppress the EEGLAB-like input dialog.
        renderer: Optional dialog renderer for tests.
        return_com: Return ``(figures, command)`` when true.
        **kwargs: Additional ``topoplot`` options such as ``electrodes``,
            ``colorbar`` and ``maplimits``.
    """
    if EEG is None:
        return ([], "") if return_com else []

    typeplot = int(typeplot)
    if gui is None:
        gui = items is None
    if typeplot not in {0, 1}:
        raise ValueError("typeplot must be 1 for ERP maps or 0 for component maps")
    _validate_topoplot_inputs(EEG, typeplot)
    if gui:
        result = _run_gui(EEG, typeplot=typeplot, renderer=renderer)
        if result is None:
            return ([], "") if return_com else []
        items = result["items"]
        topotitle = result["topotitle"]
        rowcols = result["rowcols"]
        plotdip = result["plotdip"]
        kwargs = {**kwargs, **result["options"]}
    elif items is None:
        raise ValueError("items must be provided when gui=False")

    option_args = args
    if plotdip is not None and not _is_plotdip_value(plotdip):
        option_args = (plotdip, *args)
        plotdip = 0
    options = parse_key_value_args(option_args, kwargs, lowercase_kwargs=True)
    options = _normalise_topoplot_options(EEG, options)
    items_array = _parse_numeric_sequence(items)
    if items_array.size == 0:
        raise ValueError("Nothing to plot; provide at least one latency or component index")
    rowcols_array = _normalise_rowcols(rowcols, len(items_array))
    topotitle = str(EEG.get("setname") or "") if topotitle is None else str(topotitle)

    if typeplot == 1:
        maps, labels = _erp_maps(EEG, items_array)
    elif typeplot == 0:
        maps, labels = _component_maps(EEG, items_array)
    else:
        raise ValueError("typeplot must be 1 for ERP maps or 0 for component maps")

    figures = _plot_map_pages(
        maps,
        labels,
        chanlocs_as_list(EEG.get("chanlocs", [])),
        topotitle=topotitle,
        rowcols=rowcols_array,
        options=dict(options),
        typeplot=typeplot,
    )
    command = _history_command(typeplot, items_array, topotitle, rowcols_array, int(bool(plotdip)), options)
    return (figures, command) if return_com else figures


def pop_topoplot_dialog_spec(EEG: dict[str, Any], *, typeplot: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_topoplot``."""
    title = (
        "Plot ERP scalp maps in 2-D -- pop_topoplot()"
        if int(typeplot)
        else "Plot component scalp maps in 2-D -- pop_topoplot()"
    )
    if int(typeplot):
        first_label = "Plotting ERP scalp maps at these latencies"
        range_label = f"(range: {round(float(EEG.get('xmin', 0)) * 1000)} to {round(float(EEG.get('xmax', 0)) * 1000)} ms, NaN -> empty):"
        item_value = ""
    else:
        first_label = "Component numbers"
        range_label = "(negate index to invert component polarity; NaN -> empty subplot; Ex: -1 NaN 3)"
        n_components = np.asarray(EEG.get("icaweights", [])).shape[0]
        item_value = f"1:{n_components}" if n_components else ""

    controls = [
        ControlSpec("text", first_label),
        ControlSpec("edit", tag="items", value=item_value),
        ControlSpec("text", range_label),
        ControlSpec("spacer"),
        ControlSpec("text", "Plot title"),
        ControlSpec("edit", tag="topotitle", value=str(EEG.get("setname") or "")),
        ControlSpec("text", "Plot geometry (rows,col.); [] -> near square"),
        ControlSpec("edit", tag="rowcols", value="[]"),
    ]
    if not int(typeplot):
        controls.extend(
            [
                ControlSpec("text", "Plot associated dipole(s) (if present)"),
                ControlSpec("checkbox", tag="plotdip", value=False),
                ControlSpec("spacer"),
                ControlSpec("spacer"),
            ]
        )
    controls.extend(
        [
            ControlSpec("spacer"),
            ControlSpec(
                "text",
                f"-> Additional topoplot(){' (and dipole)' if not int(typeplot) else ''} options (see Help)",
            ),
            ControlSpec("edit", tag="options", value=_default_options_text(EEG)),
        ]
    )
    geometry: tuple[Any, ...] = (
        (1.5, 1),
        1,
        1,
        (1.5, 1),
        (1.5, 1),
        (1.55, 0.2, 0.8),
        1,
        1,
        1,
    )
    if int(typeplot):
        geometry = geometry[:5] + geometry[6:]
        geomvert = (1, 1, 0.8, 1, 1, 0.5, 1, 1)
        size = (645, 404)
    else:
        geomvert = (1, 1, 0.5, 1, 1, 1, 0.2, 1, 1)
        size = (697, 440)
    return DialogSpec(
        title=title,
        controls=tuple(controls),
        geometry=geometry,
        function_name="pop_topoplot",
        eeglab_source="functions/popfunc/pop_topoplot.m",
        help_text="pophelp('pop_topoplot')",
        size=size,
        geomvert=geomvert,
    )


def plot_channel_locations(EEG: dict[str, Any], *, mode: str = "labels", return_com: bool = False):
    """Plot channel locations by name or by number, matching EEGLAB menu actions.

    The returned command is valid ``eegprep-console`` Python, matching the
    Python-callable history shape used by ``pop_topoplot``.
    """
    chanlocs = chanlocs_as_list(EEG.get("chanlocs", []))
    if not chanlocs:
        raise ValueError("cannot plot topography without channel location file")
    electrodes = "numpoint" if mode == "numbers" else "labelpoint"
    fig, *_ = topoplot([], chanlocs, style="blank", electrodes=electrodes, title="Channel locations")
    command = f"topoplot([], EEG['chanlocs'], style='blank', electrodes={electrodes!r})"
    return (fig, command) if return_com else fig


def _run_gui(EEG: dict[str, Any], *, typeplot: int, renderer: Any | None = None) -> dict[str, Any] | None:
    spec = pop_topoplot_dialog_spec(EEG, typeplot=typeplot)
    result = inputgui(spec, renderer=renderer)
    if not result:
        return None
    options_text = str(result.get("options", "") or "")
    return {
        "items": _parse_items_text(str(result.get("items", "") or "")),
        "topotitle": str(result.get("topotitle", "") or ""),
        "rowcols": _parse_rowcols_text(str(result.get("rowcols", "") or "")),
        "plotdip": bool(result.get("plotdip", False)),
        "options": _parse_options_text(options_text),
    }


def _plot_map_pages(
    maps: list[np.ndarray | None],
    labels: list[str],
    chanlocs: list[dict[str, Any]],
    *,
    topotitle: str,
    rowcols: tuple[int, int],
    options: dict[str, Any],
    typeplot: int,
) -> list[Any]:
    rows, cols = rowcols
    per_page = rows * cols
    figures = []
    colorbar = _is_on(options.pop("colorbar", "on"))
    maplimits = options.pop("maplimits", _default_maplimits(maps, typeplot))
    for page_start in range(0, len(maps), per_page):
        page_maps = maps[page_start : page_start + per_page]
        page_labels = labels[page_start : page_start + per_page]
        plotted_map_count = sum(values is not None for values in page_maps)
        fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(cols * 2.1, rows * 2.0))
        colorbar_image = None
        plotted_axes = []
        for ax, values, label in zip(axes.ravel(), page_maps, page_labels):
            if values is None:
                ax.axis("off")
                continue
            topoplot(
                values,
                chanlocs,
                axes=ax,
                colorbar=colorbar and plotted_map_count == 1,
                maplimits=maplimits,
                **options,
            )
            if ax.images:
                colorbar_image = ax.images[-1]
                plotted_axes.append(ax)
            ax.set_title(label)
        for ax in axes.ravel()[len(page_maps) :]:
            ax.axis("off")
        if topotitle:
            fig.suptitle(topotitle, fontweight="bold")
        fig.tight_layout()
        if colorbar and plotted_map_count > 1 and colorbar_image is not None:
            fig.colorbar(colorbar_image, ax=plotted_axes, shrink=0.7)
        figures.append(fig)
    return figures


def _erp_maps(EEG: dict[str, Any], latencies_ms: np.ndarray) -> tuple[list[np.ndarray | None], list[str]]:
    _require_chanlocs(EEG)
    data = np.asarray(EEG.get("data"))
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    if data.ndim != 3:
        raise ValueError("EEG.data must be a 2-D or 3-D channel-major array")
    positions = _latency_positions(EEG, latencies_ms)
    maps = []
    labels = []
    for latency, position in zip(latencies_ms, positions):
        if np.isnan(latency):
            maps.append(None)
            labels.append("")
            continue
        maps.append(np.nanmean(data[:, position, :], axis=1))
        labels.append(f"{int(round(float(latency)))} ms")
    return maps, labels


def _component_maps(EEG: dict[str, Any], components: np.ndarray) -> tuple[list[np.ndarray | None], list[str]]:
    _require_chanlocs(EEG)
    _require_ica(EEG)
    icawinv = np.asarray(EEG.get("icawinv", []), dtype=float)
    maps = []
    labels = []
    for component in components:
        if np.isnan(component):
            maps.append(None)
            labels.append("")
            continue
        index = int(abs(component))
        if index < 1 or index > icawinv.shape[1]:
            raise ValueError(f"component index {index} is outside available ICA components")
        values = icawinv[:, index - 1]
        maps.append(-values if component < 0 else values)
        # EEGLAB titles inverted-polarity maps with the negative component index.
        labels.append(f"IC {int(component)}")
    return maps, labels


def _latency_positions(EEG: dict[str, Any], latencies_ms: np.ndarray) -> np.ndarray:
    pnts = int(EEG.get("pnts", 0))
    xmin = float(EEG.get("xmin", 0))
    xmax = float(EEG.get("xmax", xmin))
    if pnts <= 0:
        raise ValueError("EEG.pnts must be positive")
    positions = np.zeros_like(latencies_ms, dtype=int)
    finite = np.isfinite(latencies_ms)
    invalid = ~finite & ~np.isnan(latencies_ms)
    if np.any(invalid):
        raise ValueError("requested latency is outside the epoch time range")
    if xmax == xmin and np.any(finite):
        raise ValueError("requested latency is outside the epoch time range")
    if xmax != xmin and np.any(finite):
        positions[finite] = np.rint(((latencies_ms[finite] / 1000.0) - xmin) / (xmax - xmin) * (pnts - 1)).astype(int)
    valid = np.isnan(latencies_ms) | ((positions >= 0) & (positions < pnts))
    if not np.all(valid):
        raise ValueError("requested latency is outside the epoch time range")
    positions[np.isnan(latencies_ms)] = 0
    return positions


def _normalise_topoplot_options(EEG: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    normalised = {str(key).lower(): value for key, value in options.items()}
    normalised.setdefault("electrodes", "off" if int(EEG.get("nbchan", 0) or 0) > 64 else "on")
    return normalised


def _default_options_text(EEG: dict[str, Any]) -> str:
    return "'electrodes', 'off'" if int(EEG.get("nbchan", 0) or 0) > 64 else "'electrodes', 'on'"


def _default_maplimits(maps: list[np.ndarray | None], typeplot: int) -> str | list[float]:
    if not int(typeplot):
        return "absmax"
    finite_maps = [np.asarray(values).ravel() for values in maps if values is not None]
    if not finite_maps:
        return [-1.0, 1.0]
    finite_values = np.concatenate(finite_maps)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return [-1.0, 1.0]
    limit = float(np.max(np.abs(finite_values)))
    return [-limit, limit] if limit else [-1.0, 1.0]


def _normalise_rowcols(rowcols: Any, count: int) -> tuple[int, int]:
    values = _parse_numeric_sequence(rowcols)
    if values.size == 0 or int(values[0]) == 0:
        cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
        return rows, cols
    if values.size != 2:
        raise ValueError("rowcols must be [] or [rows, columns]")
    rows, cols = int(values[0]), int(values[1])
    if rows < 1 or cols < 1:
        raise ValueError("rowcols values must be positive")
    return rows, cols


def _parse_items_text(text: str) -> list[float]:
    stripped = re.sub(r"\s*:\s*", ":", text.strip().strip("[]"))
    if not stripped:
        return []
    values = []
    for token in re.split(r"[\s,]+", stripped):
        if not token:
            continue
        if ":" in token:
            values.extend(_parse_colon_sequence(token))
        else:
            values.append(_parse_float_token(token))
    return values


def _parse_colon_sequence(token: str) -> list[float]:
    pieces = token.split(":")
    if len(pieces) not in {2, 3}:
        raise ValueError(f"Invalid colon range: {token}")
    start = _parse_float_token(pieces[0])
    if len(pieces) == 2:
        stop = _parse_float_token(pieces[1])
        step = 1.0 if stop >= start else -1.0
    else:
        step = _parse_float_token(pieces[1])
        stop = _parse_float_token(pieces[2])
    if step == 0 or not np.all(np.isfinite([start, step, stop])):
        raise ValueError(f"Invalid colon range: {token}")
    if (stop - start) * step < 0:
        return []
    values = []
    current = start
    tolerance = abs(step) * 1e-10
    if step > 0:
        while current <= stop + tolerance:
            values.append(float(current))
            current += step
    else:
        while current >= stop - tolerance:
            values.append(float(current))
            current += step
    return values


def _parse_rowcols_text(text: str) -> list[float]:
    return _parse_numeric_sequence(text).tolist()


def _parse_numeric_sequence(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float).ravel()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).ravel()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([value], dtype=float)
    text = str(value).strip().strip("[]")
    if not text:
        return np.asarray([], dtype=float)
    return np.asarray([_parse_float_token(token) for token in re.split(r"[\s,]+", text) if token], dtype=float)


def _parse_options_text(text: str) -> dict[str, Any]:
    tokens = parse_text_tokens(text)
    if not tokens:
        return {}
    if len(tokens) % 2:
        raise ValueError("Additional topoplot options must be key/value pairs")
    return {str(tokens[index]).lower(): _parse_option_value(tokens[index + 1]) for index in range(0, len(tokens), 2)}


def _parse_option_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            return _parse_numeric_sequence(text).tolist()
        try:
            return _parse_float_token(text)
        except ValueError:
            return text
    return value


def _parse_float_token(token: str) -> float:
    text = str(token).strip()
    if text.lower() == "nan":
        return float("nan")
    if text.lower() == "inf":
        return float("inf")
    if text.lower() == "-inf":
        return float("-inf")
    return float(text)


def _require_chanlocs(EEG: dict[str, Any]) -> None:
    if not chanlocs_as_list(EEG.get("chanlocs", [])):
        raise ValueError("cannot plot topography without channel location file")


def _require_ica(EEG: dict[str, Any]) -> None:
    icawinv = EEG.get("icawinv", [])
    if icawinv is None or np.asarray(icawinv).size == 0:
        raise ValueError("no ICA data for this set, first run ICA")


def _validate_topoplot_inputs(EEG: dict[str, Any], typeplot: int) -> None:
    _require_chanlocs(EEG)
    if typeplot == 0:
        _require_ica(EEG)


def _is_on(value: Any) -> bool:
    return str(value).lower() in {"on", "yes", "true", "1"}


def _is_plotdip_value(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, np.integer, np.floating))


def _history_command(
    typeplot: int,
    items: np.ndarray,
    topotitle: str,
    rowcols: tuple[int, int],
    plotdip: int,
    options: dict[str, Any],
) -> str:
    pieces = [
        "EEG",
        f"typeplot={int(typeplot)}",
        f"items={_python_literal(items)}",
        f"topotitle={_python_literal(topotitle)}",
        f"rowcols={_python_literal(list(rowcols))}",
        f"plotdip={int(plotdip)}",
    ]
    for key, value in options.items():
        pieces.append(f"{key}={_python_literal(value)}")
    return f"pop_topoplot({', '.join(pieces)})"


def _python_literal(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return "float('nan')"
        if np.isposinf(value):
            return "float('inf')"
        if np.isneginf(value):
            return "float('-inf')"
        if value.is_integer():
            return str(int(value))
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(_python_literal(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    return repr(value)


__all__ = ["plot_channel_locations", "pop_topoplot", "pop_topoplot_dialog_spec"]
