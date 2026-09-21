"""Matplotlib equivalent of EEGLAB's recursive ``setfont`` helper."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text


def setfont(handle: Figure | Axes, *args: Any, handletype: str | None = None, **kwargs: Any) -> Figure | Axes:
    """Apply font properties to text in a Matplotlib figure or axes.

    ``handletype`` may be ``"xlabels"``, ``"ylabels"``, ``"titles"``,
    ``"axis"``, or ``"strings"``. EEGLAB-style positional name/value pairs
    and Python keyword properties are both accepted.
    """
    positional = list(args)
    if positional and str(positional[0]).lower() == "handletype":
        if len(positional) < 2:
            raise TypeError("handletype requires a value")
        if handletype is not None:
            raise TypeError("handletype was supplied twice")
        handletype = str(positional[1])
        positional = positional[2:]
    if len(positional) % 2:
        raise TypeError("font properties must be name/value pairs")
    properties = dict(kwargs)
    for index in range(0, len(positional), 2):
        name = positional[index]
        if not isinstance(name, str):
            raise TypeError("font property names must be strings")
        properties[name] = positional[index + 1]
    normalized = {_property_name(name): value for name, value in properties.items()}
    axes = _axes(handle)
    category = "" if handletype is None else handletype.lower()
    if category not in {"", "xlabels", "ylabels", "titles", "axis", "strings"}:
        raise ValueError(f"unrecognized handletype: {handletype}")

    if category == "xlabels":
        targets: list[Any] = [axis.xaxis.label for axis in axes]
    elif category == "ylabels":
        targets = [axis.yaxis.label for axis in axes]
    elif category == "titles":
        targets = [axis.title for axis in axes]
    elif category == "axis":
        targets = axes
    elif category == "strings":
        targets = _all_text(handle)
    else:
        targets = [*axes, *_all_text(handle)]

    for target in _unique_objects(targets):
        if isinstance(target, Axes):
            _set_axis_font(target, normalized)
        elif isinstance(target, Text):
            target.set(**normalized)
    return handle


def _axes(handle: Figure | Axes) -> list[Axes]:
    if isinstance(handle, Figure):
        return list(handle.axes)
    if isinstance(handle, Axes):
        return [handle]
    raise TypeError("handle must be a Matplotlib Figure or Axes")


def _all_text(handle: Figure | Axes) -> list[Text]:
    return list(handle.findobj(match=Text))


def _unique_objects(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        identifier = id(value)
        if identifier not in seen:
            seen.add(identifier)
            result.append(value)
    return result


def _property_name(name: str) -> str:
    normalized = name.replace("_", "").lower()
    aliases = {
        "fontsize": "fontsize",
        "fontname": "fontfamily",
        "fontfamily": "fontfamily",
        "fontweight": "fontweight",
        "fontstyle": "fontstyle",
        "color": "color",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported font property: {name}")
    return aliases[normalized]


def _set_axis_font(axis: Axes, properties: dict[str, Any]) -> None:
    text_properties = {name: value for name, value in properties.items() if name != "color"}
    color = properties.get("color")
    tick_properties: dict[str, Any] = {}
    if "fontsize" in text_properties:
        tick_properties["labelsize"] = text_properties.pop("fontsize")
    if color is not None:
        tick_properties["colors"] = color
    axis.tick_params(axis="both", **tick_properties)
    for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        label.set(**text_properties)
    if color is not None:
        axis.xaxis.label.set_color(color)
        axis.yaxis.label.set_color(color)
        for spine in axis.spines.values():
            spine.set_color(color)


__all__ = ["setfont"]
