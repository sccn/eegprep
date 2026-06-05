"""Pair corresponding channels between two channel-location structures."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args, parse_numeric_sequence


FIDUCIAL_ALIASES = (("nz", "nasion", "fidnz"), ("lpa", "left", "fidt10"), ("rpa", "right", "fidt9"))


def pop_chancoresp(chans1: Any, chans2: Any, *args: Any, return_com: bool = False, **kwargs: Any) -> Any:
    """Return 1-based channel correspondences by label."""
    if isinstance(chans1, str):
        result = _subcommand(chans1, chans2, *args)
        return (*result, "") if return_com else result
    options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
    labels1 = _labels(chans1)
    labels2 = _labels(chans2)
    chanlist1 = _indices_option(options.get("chanlist1"))
    chanlist2 = _indices_option(options.get("chanlist2"))
    if len(chanlist1) != len(chanlist2):
        raise ValueError("input arguments 'chanlist1' and 'chanlist2' must have the same length")
    if not chanlist1:
        autoselect = str(options.get("autoselect", "all")).lower()
        if autoselect == "all":
            chanlist1, chanlist2 = _auto_all(labels1, labels2)
        elif autoselect == "fiducials":
            chanlist1, chanlist2 = _auto_fiducials(labels1, labels2)
        elif autoselect != "none":
            raise ValueError(f"Unsupported pop_chancoresp autoselect mode: {autoselect}")
    command = _history_command(options)
    return (chanlist1, chanlist2, command) if return_com else (chanlist1, chanlist2)


def _subcommand(command: str, *args: Any) -> tuple[Any, ...]:
    name = command.lower()
    if name == "clear":
        labels1 = _labels(args[0])
        labels2 = _labels(args[1])
        return _list_text(labels1, labels2, [], [])
    if name == "auto":
        labels1 = _labels(args[0])
        labels2 = _labels(args[1])
        chanlist1, chanlist2 = _auto_all(labels1, labels2)
        return (*_list_text(labels1, labels2, chanlist1, chanlist2), chanlist1, chanlist2)
    if name in {"pair", "unpair"}:
        ind1 = int(args[0])
        ind2 = int(args[1])
        labels1 = _labels(args[2])
        labels2 = _labels(args[3])
        chanlist1 = list(args[4]) if len(args) > 4 else []
        chanlist2 = list(args[5]) if len(args) > 5 else []
        if name == "pair":
            if ind1 not in chanlist1 and ind2 not in chanlist2:
                chanlist1.append(ind1)
                chanlist2.append(ind2)
        else:
            for pos, (left, right) in reversed(list(enumerate(zip(chanlist1, chanlist2)))):
                if left == ind1 or right == ind2:
                    chanlist1.pop(pos)
                    chanlist2.pop(pos)
        return (*_list_text(labels1, labels2, chanlist1, chanlist2), chanlist1, chanlist2)
    raise ValueError(f"Unsupported pop_chancoresp subcommand: {command}")


def _labels(chans: Any) -> list[str]:
    if isinstance(chans, dict) and "label" in chans:
        labels = chans["label"]
        return [str(label) for label in labels]
    if isinstance(chans, dict) and "chanlocs" in chans:
        chans = chans["chanlocs"]
    if isinstance(chans, (list, tuple)) and all(isinstance(item, str) for item in chans):
        return [str(item) for item in chans]
    return [str(loc.get("labels", loc.get("label", ""))) for loc in chanlocs_as_list(chans)]


def _auto_all(labels1: list[str], labels2: list[str]) -> tuple[list[int], list[int]]:
    lower2 = [label.lower() for label in labels2]
    chanlist1 = []
    chanlist2 = []
    for index, label in enumerate(labels1, start=1):
        try:
            match = lower2.index(label.lower()) + 1
        except ValueError:
            continue
        chanlist1.append(index)
        chanlist2.append(match)
    return chanlist1, chanlist2


def _auto_fiducials(labels1: list[str], labels2: list[str]) -> tuple[list[int], list[int]]:
    lower1 = [label.lower() for label in labels1]
    lower2 = [label.lower() for label in labels2]
    chanlist1 = []
    chanlist2 = []
    for aliases in FIDUCIAL_ALIASES:
        left = _first_alias(lower1, aliases)
        right = _first_alias(lower2, aliases)
        if left is not None and right is not None:
            chanlist1.append(left + 1)
            chanlist2.append(right + 1)
    return chanlist1, chanlist2


def _first_alias(labels: list[str], aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        if alias in labels:
            return labels.index(alias)
    return None


def _indices_option(value: Any) -> list[int]:
    if value in (None, "", []):
        return []
    return [int(item) for item in parse_numeric_sequence(value, dtype=int)]


def _list_text(
    labels1: list[str], labels2: list[str], chanlist1: list[int], chanlist2: list[int]
) -> tuple[list[str], list[str]]:
    paired = dict(zip(chanlist1, chanlist2))
    reverse = dict(zip(chanlist2, chanlist1))
    left = [
        f"{index:2d} - {label:3s}   -> {paired[index]:2d} - {labels2[paired[index] - 1]:3s}"
        if index in paired
        else f"{index:2d} - {label:3s}"
        for index, label in enumerate(labels1, start=1)
    ]
    right = [
        f"{index:2d} - {label:3s}   -> {reverse[index]:2d} - {labels1[reverse[index] - 1]:3s}"
        if index in reverse
        else f"{index:2d} - {label:3s}"
        for index, label in enumerate(labels2, start=1)
    ]
    return left, right


def _history_command(options: dict[str, Any]) -> str:
    pieces = []
    for key, value in options.items():
        pieces.append(format_history_value(key))
        pieces.append(format_history_value(value, cell_for_sequence="any_strings"))
    suffix = ", " + ", ".join(pieces) if pieces else ""
    return f"[chanlist1, chanlist2] = pop_chancoresp(EEG['chanlocs'], ref_locs{suffix});"


__all__ = ["pop_chancoresp"]
