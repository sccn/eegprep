"""Write a Matplotlib figure as a MATLAB ``.fig`` file.

MATLAB ``openfig`` reads a MAT-file whose ``hgS_070000`` variable is the HG1
graphics tree (figure, axes, line, image, text). Current MATLAB also stores an
MCOS ``hgM_070000`` blob; scipy cannot write that, and ``openfig`` still loads
the ``hgS_070000`` tree from older ``.fig`` files.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import BinaryIO

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps, rcParams
from matplotlib.backend_bases import register_backend
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import NoNorm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.io import savemat

_NODE = np.dtype(
    [
        ("type", "O"),
        ("handle", "O"),
        ("properties", "O"),
        ("children", "O"),
        ("special", "O"),
    ]
)
_MATLAB_MARKERS = set("os+x*.d^v<>ph")
_MARKER_ALIASES = {
    "none": "none",
    "None": "none",
    "": "none",
    " ": "none",
    ",": ".",
    "1": "v",
    "2": "^",
    "3": "<",
    "4": ">",
    "H": "h",
    "X": "x",
    "D": "d",
    "P": "p",
    "8": "o",
    "|": "none",
    "_": "none",
}
_LINE_STYLES = {
    "-": "-",
    "--": "--",
    "-.": "-.",
    ":": ":",
    "solid": "-",
    "dashed": "--",
    "dashdot": "-.",
    "dotted": ":",
    "": "none",
    " ": "none",
    "none": "none",
    "None": "none",
}
_VERTICAL = {
    "center": "middle",
    "center_baseline": "baseline",
    "top": "top",
    "bottom": "bottom",
    "baseline": "baseline",
    "middle": "middle",
}


def savefig(figure: Figure, filename: str | os.PathLike[str] | BinaryIO) -> str:
    """Save ``figure`` as a MATLAB ``.fig`` file.

    The file stores the figure, each axes, and the lines, images, and text on
    those axes (data, limits, colors, scales, and labels). Title and axis
    labels are text children indexed by the axes ``special`` vector, which is
    how ``openfig`` attaches them. Scatter, bars, patches, and legends are
    omitted. A figure has one colormap, taken from the first image.
    ``Interpreter`` is ``none`` so label text is stored literally. Characters
    outside ASCII are replaced (a Unicode minus becomes ``-``) so each MAT
    character array is one byte per character. Mathtext tick labels are stored
    as the plain tick value. Importing this module also
    registers Matplotlib's ``fig`` format, so ``figure.savefig("name.fig")``
    uses the same writer.

    Args:
        figure: Matplotlib figure to export.
        filename: Destination path. A ``.fig`` suffix is added when missing.
            A binary file object is written as-is.

    Returns:
        The path written, or the file object's name when ``filename`` is a
        file object.
    """
    if not isinstance(figure, Figure):
        raise TypeError("savefig expected a matplotlib Figure")
    destination, returned = _destination(filename)
    _ensure_drawn(figure)
    savemat(
        destination,
        {"hgS_070000": _figure_node(figure, iter(_handle_numbers()))},
        format="5",
        do_compression=False,
        long_field_names=True,
    )
    return returned


def register_matlab_fig_format() -> None:
    """Register Matplotlib's ``fig`` format so ``Figure.savefig`` writes a ``.fig``."""
    register_backend("fig", _FigureCanvasMatlabFig, "MATLAB figure")


class _FigureCanvasMatlabFig(FigureCanvasAgg):
    def print_fig(self, filename, **_kwargs):
        savefig(self.figure, filename)


def _destination(filename: str | os.PathLike[str] | BinaryIO) -> tuple[str | BinaryIO, str]:
    if hasattr(filename, "write"):
        return filename, str(getattr(filename, "name", "") or "")
    if isinstance(filename, str) and not filename:
        raise ValueError("savefig filename is empty")
    path = os.fspath(filename)
    if not path.lower().endswith(".fig"):
        path += ".fig"
    return path, path


def _ensure_drawn(figure: Figure) -> None:
    if figure.canvas is None:
        FigureCanvasAgg(figure)
    figure.canvas.draw()


def _handle_numbers() -> Iterator[float]:
    handle = 1.0
    while True:
        yield handle
        handle += 1.0


def _empty() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float64)


def _row(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def _column(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def _ascii(value: str) -> str:
    # scipy writes mxCHAR as UTF-8 while the array size counts characters.
    # A multibyte character makes those two disagree and MATLAB's fig loader
    # aborts. U+2212 is the minus Matplotlib puts in tick labels.
    return value.replace("\u2212", "-").encode("ascii", "replace").decode("ascii")


def _scalar(value: float) -> np.ndarray:
    return np.array([[float(value)]], dtype=np.float64)


def _node(
    kind: str,
    handle: float,
    properties: dict,
    children: list[np.ndarray],
    special: np.ndarray | None = None,
) -> np.ndarray:
    node = np.empty((1, 1), dtype=_NODE)
    node[0, 0]["type"] = kind
    node[0, 0]["handle"] = _scalar(handle)
    node[0, 0]["properties"] = properties
    node[0, 0]["children"] = _stack(children)
    # Axes ``special`` is a 4x1 index vector (title, xlabel, ylabel, zlabel).
    # Other nodes leave it empty, matching hgsave.
    node[0, 0]["special"] = _empty() if special is None else special
    return node


def _stack(nodes: list[np.ndarray]) -> np.ndarray:
    if not nodes:
        return _empty()
    stacked = np.empty((1, len(nodes)), dtype=_NODE)
    for index, node in enumerate(nodes):
        stacked[0, index] = node[0, 0]
    return stacked


def _figure_node(figure: Figure, handles: Iterator[float]) -> np.ndarray:
    name = _figure_name(figure)
    properties: dict = {
        "Color": _color(figure.get_facecolor()),
        "Position": _row((100.0, 100.0, *(np.asarray(figure.get_size_inches()) * float(figure.dpi)))),
        "Units": "pixels",
        "NumberTitle": "off" if name else "on",
        "Colormap": _colormap(figure),
        "InvertHardcopy": "off",
        "Visible": "on",
    }
    if name:
        properties["Name"] = _ascii(name)
    handle = next(handles)
    children = [_axes_node(ax, handles) for ax in figure.axes]
    for text in figure.texts:
        node = _text_node(text, handles, None)
        if node is not None:
            children.append(node)
    return _node("figure", handle, properties, children)


def _figure_name(figure: Figure) -> str:
    manager = figure.canvas.manager
    title = manager.get_window_title() if manager is not None else ""
    title = title or figure.get_label() or ""
    if title == f"Figure {figure.number}":
        return ""
    return title


def _colormap(figure: Figure) -> np.ndarray:
    image = next((candidate for ax in figure.axes for candidate in ax.images), None)
    cmap = image.get_cmap() if image is not None else colormaps[rcParams["image.cmap"]]
    count = max(2, min(int(cmap.N), 256))
    return np.asarray(cmap(np.linspace(0.0, 1.0, count)), dtype=np.float64)[:, :3]


def _axes_node(ax, handles: Iterator[float]) -> np.ndarray:
    handle = next(handles)
    children: list[np.ndarray] = []
    children.extend(_line_node(line, handles) for line in ax.lines)
    children.extend(_image_node(image, handles) for image in ax.images)
    for text in ax.texts:
        node = _text_node(text, handles, ax)
        if node is not None:
            children.append(node)
    # hgsave always records four label texts and points ``special`` at them.
    # A 0 index makes the HG1 converter read off the front of the child list.
    label_artists = [ax.title, ax.xaxis.label, ax.yaxis.label]
    label_artists.append(ax.zaxis.label if ax.name == "3d" else None)
    special: list[float] = []
    for artist in label_artists:
        if artist is None:
            children.append(_blank_label(handles))
        else:
            children.append(_text_node(artist, handles, ax, allow_empty=True))
        special.append(float(len(children)))
    tick_labels = ax.get_xticklabels()
    font_source = tick_labels[0] if tick_labels else ax.title
    properties: dict = {
        "Position": _row(ax.get_position().bounds),
        "Units": "normalized",
        "XLim": _limits(ax.get_xlim()),
        "YLim": _limits(ax.get_ylim()),
        "XScale": _scale(ax.xaxis),
        "YScale": _scale(ax.yaxis),
        "XDir": _direction(ax.xaxis),
        "YDir": _direction(ax.yaxis),
        "XLimMode": "manual",
        "YLimMode": "manual",
        "Box": "on" if ax.get_frame_on() and all(spine.get_visible() for spine in ax.spines.values()) else "off",
        "XGrid": _grid(ax.xaxis),
        "YGrid": _grid(ax.yaxis),
        "Color": _color(ax.get_facecolor()),
        "FontSize": _scalar(float(font_source.get_fontsize())),
        "FontName": _ascii(font_source.get_fontname()),
        "Visible": "on",
    }
    _store_ticks(properties, "X", ax.get_xticks(), [tick.get_text() for tick in ax.get_xticklabels()])
    _store_ticks(properties, "Y", ax.get_yticks(), [tick.get_text() for tick in ax.get_yticklabels()])
    if ax.images:
        clim = ax.images[0].get_clim()
        properties["CLim"] = _limits(clim)
        properties["CLimMode"] = "manual"
    if ax.name == "3d":
        properties["ZLim"] = _limits(ax.get_zlim())
        properties["ZScale"] = _scale(ax.zaxis)
        properties["ZDir"] = _direction(ax.zaxis)
        properties["ZGrid"] = _grid(ax.zaxis)
    return _node("axes", handle, properties, children, _column(special))


def _line_node(line, handles: Iterator[float]) -> np.ndarray:
    x = np.ma.filled(np.ma.asarray(line.get_xdata(), dtype=np.float64), np.nan).ravel()
    y = np.ma.filled(np.ma.asarray(line.get_ydata(), dtype=np.float64), np.nan).ravel()
    color = _artist_color(line.get_color())
    marker = _marker(line.get_marker())
    properties: dict = {
        "XData": _row(x),
        "YData": _row(y),
        "Color": color,
        "LineStyle": _LINE_STYLES.get(line.get_linestyle(), "none"),
        "LineWidth": _scalar(float(line.get_linewidth())),
        "Marker": marker,
        "MarkerSize": _scalar(float(line.get_markersize())),
    }
    if isinstance(line, Line3D):
        z_values = np.ma.filled(np.ma.asarray(line.get_data_3d()[2], dtype=np.float64), np.nan).ravel()
        if z_values.size == x.size and z_values.size:
            properties["ZData"] = _row(z_values)
    if marker != "none":
        # MATLAB MarkerFaceColor 'auto' is the axes color. Matplotlib 'auto' is the line color.
        properties["MarkerEdgeColor"] = _resolved_marker_color(line, line.get_markeredgecolor(), color)
        properties["MarkerFaceColor"] = _resolved_marker_color(line, line.get_markerfacecolor(), color)
    label = line.get_label()
    if label and not str(label).startswith("_"):
        properties["DisplayName"] = _ascii(str(label))
    return _node("graph2d.lineseries", next(handles), properties, [])


def _image_node(image, handles: Iterator[float]) -> np.ndarray:
    data = np.ma.filled(np.ma.asarray(image.get_array()), np.nan)
    if data.ndim == 3 and data.shape[-1] == 4:
        data = data[..., :3]
    if data.ndim != 3 or np.issubdtype(data.dtype, np.floating):
        data = np.asarray(data, dtype=np.float64)
    rows, cols = data.shape[:2]
    xdata, ydata = _image_coordinates(image, rows, cols)
    mapping = "direct" if isinstance(image.norm, NoNorm) else "scaled"
    properties = {
        "CData": np.asarray(data),
        "XData": xdata,
        "YData": ydata,
        "CDataMapping": mapping,
    }
    return _node("image", next(handles), properties, [])


def _image_coordinates(image, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    # Matplotlib extent is the outer edge of the image. MATLAB XData/YData are
    # the centers of the first and last pixels.
    left, right, bottom, top = image.get_extent()

    def centers(start: float, stop: float, count: int) -> tuple[float, float]:
        if count <= 1:
            mid = (start + stop) / 2.0
            return mid, mid
        half = (stop - start) / count / 2.0
        return start + half, stop - half

    x0, x1 = centers(left, right, cols)
    if image.origin == "upper":
        y0, y1 = centers(top, bottom, rows)
    else:
        y0, y1 = centers(bottom, top, rows)
    return _row((x0, x1)), _row((y0, y1))


def _text_node(text, handles: Iterator[float], ax, allow_empty: bool = False) -> np.ndarray | None:
    content = _ascii(str(text.get_text() or ""))
    if not content and not allow_empty:
        return None
    if ax is not None and text.get_transform() == ax.transData:
        position = np.asarray(text.get_position(), dtype=float).reshape(-1)
        units = "data"
    else:
        target = text.figure.transFigure if ax is None else ax.transAxes
        display = np.asarray(text.get_transform().transform(text.get_position()), dtype=float)
        position = np.asarray(target.inverted().transform(display), dtype=float)
        units = "normalized"
    if position.size == 2:
        position = np.array([position[0], position[1], 0.0])
    weight = text.get_fontweight()
    bold = weight in {"bold", "heavy", "semibold", "demibold"} or (
        isinstance(weight, (int, float)) and float(weight) >= 600
    )
    properties = {
        "String": _matlab_text(content),
        "Position": _row(position[:3]),
        "Units": units,
        "Color": _artist_color(text.get_color()),
        "FontSize": _scalar(float(text.get_fontsize())),
        "FontName": _ascii(text.get_fontname()),
        "FontWeight": "bold" if bold else "normal",
        "FontAngle": "italic" if text.get_fontstyle() == "italic" else "normal",
        "HorizontalAlignment": text.get_ha() if text.get_ha() in {"left", "center", "right"} else "left",
        "VerticalAlignment": _VERTICAL.get(text.get_va(), "middle"),
        "Rotation": _scalar(float(text.get_rotation())),
        "Interpreter": "none",
    }
    return _node("text", next(handles), properties, [])


def _blank_label(handles: Iterator[float]) -> np.ndarray:
    properties = {
        "String": "",
        "Position": _row((0.0, 0.0, 0.0)),
        "Units": "data",
        "Color": _row((0.0, 0.0, 0.0)),
        "FontSize": _scalar(10.0),
        "FontName": "Helvetica",
        "FontWeight": "normal",
        "FontAngle": "normal",
        "HorizontalAlignment": "center",
        "VerticalAlignment": "middle",
        "Rotation": _scalar(0.0),
        "Interpreter": "none",
    }
    return _node("text", next(handles), properties, [])


def _matlab_text(content: str):
    # A 1-D array of fixed-width strings becomes a 2-D mxCHAR (one row per
    # line). A 2-D array of single characters becomes 3-D, which openfig rejects.
    lines = content.split("\n")
    if len(lines) == 1:
        return lines[0]
    width = max(len(line) for line in lines)
    return np.array([line.ljust(width) for line in lines], dtype=f"U{width}")


def _store_ticks(properties: dict, prefix: str, ticks, labels: list[str]) -> None:
    values = np.asarray(ticks, dtype=np.float64).ravel()
    count = min(values.size, len(labels))
    if count == 0:
        return
    properties[f"{prefix}Tick"] = _row(values[:count])
    properties[f"{prefix}TickMode"] = "manual"
    plain = [_plain_tick_label(label, float(values[index])) for index, label in enumerate(labels[:count])]
    matrix = _tick_label_matrix(plain)
    if matrix is not None:
        properties[f"{prefix}TickLabel"] = matrix
        properties[f"{prefix}TickLabelMode"] = "manual"


def _plain_tick_label(label: str, value: float) -> str:
    text = _ascii(label).replace("\n", " ")
    if "\\" not in text and "$" not in text:
        return text
    return np.format_float_positional(value, trim="-")


def _tick_label_matrix(labels: list[str]):
    if not any(labels):
        return None
    cleaned = [_ascii(label).replace("\n", " ") for label in labels]
    width = max(len(label) for label in cleaned)
    if width == 0:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return np.array([line.ljust(width) for line in cleaned], dtype=f"U{width}")


def _limits(pair) -> np.ndarray:
    low, high = float(pair[0]), float(pair[1])
    return _row((min(low, high), max(low, high)))


def _scale(axis) -> str:
    return "log" if axis.get_scale() == "log" else "linear"


def _direction(axis) -> str:
    return "reverse" if axis.get_inverted() else "normal"


def _grid(axis) -> str:
    return "on" if any(line.get_visible() for line in axis.get_gridlines()) else "off"


def _color(value):
    if value is None or (isinstance(value, str) and value.lower() == "none"):
        return "none"
    rgba = mcolors.to_rgba(value)
    if len(rgba) > 3 and rgba[3] == 0:
        return "none"
    return _row(rgba[:3])


def _artist_color(value):
    try:
        return _color(value)
    except ValueError:
        if isinstance(value, str) and len(value) > 1 and value[0] == "C" and value[1:].isdigit():
            palette = rcParams["axes.prop_cycle"].by_key().get("color", [])
            if palette:
                return _color(palette[int(value[1:]) % len(palette)])
        raise


def _resolved_marker_color(line, value, line_color):
    if isinstance(value, str) and value.lower() == "auto":
        return line_color
    return _artist_color(value)


def _marker(marker) -> str:
    if marker is None:
        return "none"
    text = str(marker)
    if text in _MARKER_ALIASES:
        return _MARKER_ALIASES[text]
    if text in _MATLAB_MARKERS:
        return text
    return "o"


register_matlab_fig_format()
