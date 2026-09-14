"""Create axes spanning arbitrary positions in an EEGLAB-style subplot grid."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


DEFAULT_AXES_POSITION = (0.13, 0.11, 0.775, 0.815)


def sbplot(
    rows: int,
    columns: int,
    grid_position: int | tuple[int, int] | list[int],
    *properties: Any,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Create an axes in one tile or across two corner tiles.

    Unlike :func:`matplotlib.pyplot.subplot`, underlying axes are retained.
    A two-element ``grid_position`` describes opposite corners using MATLAB's
    one-based, row-major subplot numbering.

    Args:
        rows: Number of grid rows.
        columns: Number of grid columns.
        grid_position: One tile index or two opposite corner indices.
        *properties: Optional EEGLAB-style axes property/value pairs. A leading
            ``"ax", axes`` pair uses that axes' bounds as the tiling region.
        ax: Existing axes whose bounds define the tiling region.
        **kwargs: Matplotlib axes properties, such as ``facecolor``.

    Returns:
        The newly created axes.
    """
    rows, columns = int(rows), int(columns)
    if rows < 1 or columns < 1:
        raise ValueError("sbplot rows and columns must be positive")
    property_items = list(properties)
    if len(property_items) >= 2 and str(property_items[0]).lower() == "ax":
        if ax is not None and property_items[1] is not ax:
            raise ValueError("sbplot received two different parent axes")
        ax = property_items[1]
        property_items = property_items[2:]
    if len(property_items) % 2:
        raise ValueError("sbplot axes properties must be property/value pairs")

    corners = np.asarray(grid_position, dtype=int).ravel()
    if corners.size not in {1, 2}:
        raise ValueError("sbplot grid_position must contain one index or two corner indices")
    if np.any(corners < 1) or np.any(corners > rows * columns):
        raise ValueError(f"sbplot indices must be within 1..{rows * columns}")

    figure = ax.figure if ax is not None else plt.gcf()
    bounds = ax.get_position().bounds if ax is not None else DEFAULT_AXES_POSITION
    position = _grid_bounds(rows, columns, corners, bounds)
    created = figure.add_axes(position)
    options = _property_options(property_items, kwargs)
    if options:
        created.set(**options)
    return created


def _grid_bounds(
    rows: int, columns: int, corners: np.ndarray, bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    left, bottom, width, height = bounds
    if columns == 2:
        x_space = width * 0.27 / (columns - 0.27)
    else:
        x_space = 0.9 * width * 0.27 / (columns - 0.9 * 0.27)
    if rows == 2:
        y_space = height * 0.27 / (rows - 0.27)
    else:
        y_space = 0.9 * height * 0.27 / (rows - 0.9 * 0.27)
    cell_width = (width - x_space * (columns - 1)) / columns
    cell_height = (height - y_space * (rows - 1)) / rows

    row_indices = (corners - 1) // columns
    column_indices = (corners - 1) % columns
    first_column, last_column = int(np.min(column_indices)), int(np.max(column_indices))
    first_row, last_row = int(np.min(row_indices)), int(np.max(row_indices))
    column_span = last_column - first_column + 1
    row_span = last_row - first_row + 1
    lower_row = rows - last_row - 1
    return (
        left + (x_space + cell_width) * first_column,
        bottom + (y_space + cell_height) * lower_row - 0.03,
        x_space * (column_span - 1) + cell_width * column_span,
        y_space * (row_span - 1) + cell_height * row_span,
    )


def _property_options(properties: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    options = dict(kwargs)
    for index in range(0, len(properties), 2):
        options[str(properties[index])] = properties[index + 1]
    aliases = {"color": "facecolor"}
    return {aliases.get(str(key).lower(), str(key).lower()): value for key, value in options.items()}


__all__ = ["DEFAULT_AXES_POSITION", "sbplot"]
