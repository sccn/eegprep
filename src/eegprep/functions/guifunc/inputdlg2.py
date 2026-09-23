"""EEGLAB ``inputdlg2``-style text entry dialog."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec


def inputdlg2(
    prompt: str | Sequence[Any],
    title: str,
    numlines: int | Sequence[int] = 1,
    defaultanswer: Sequence[str] | None = None,
    funcname: str = "",
    *,
    renderer: Any | None = None,
) -> list[str]:
    """Prompt for one or more text values and return answers in prompt order.

    An empty list means the user cancelled the dialog. ``numlines`` is accepted
    for call compatibility; like EEGLAB's helper, EEGPrep renders one editable
    value per prompt.
    """
    spec = inputdlg2_dialog_spec(prompt, title, numlines, defaultanswer, funcname)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return []
    return [str(result[f"answer{index}"]) for index in range(len(_prompts(prompt)))]


def inputdlg2_dialog_spec(
    prompt: str | Sequence[Any],
    title: str,
    numlines: int | Sequence[int] = 1,
    defaultanswer: Sequence[str] | None = None,
    funcname: str = "",
) -> DialogSpec:
    """Build the renderer-independent specification for ``inputdlg2``."""
    prompts = _prompts(prompt)
    defaults = [""] * len(prompts) if defaultanswer is None else [str(item) for item in defaultanswer]
    if len(prompts) != len(defaults):
        raise ValueError("inputdlg2 prompts and default answers must have the same length")
    _validate_numlines(numlines, len(prompts))

    multiline = ["\n" in item for item in prompts]
    controls: list[ControlSpec] = []
    geometry: list[tuple[float, ...]] = []
    geomvert: list[float] = []
    for index, (label, default, is_multiline) in enumerate(zip(prompts, defaults, multiline)):
        controls.extend((ControlSpec("text", label), ControlSpec("edit", tag=f"answer{index}", value=default)))
        if is_multiline:
            geometry.extend(((1,), (1,)))
            geomvert.extend((max(label.count("\n") + 1, 1), 1))
        else:
            geometry.append((1, 0.6))

    return DialogSpec(
        title=title,
        controls=tuple(controls),
        geometry=tuple(geometry),
        geomvert=tuple(geomvert) or None,
        function_name=funcname or "inputdlg2",
        eeglab_source="functions/guifunc/inputdlg2.m",
        help_text=funcname or None,
        show_help_button=bool(funcname),
    )


def _prompts(prompt: str | Sequence[Any]) -> list[str]:
    raw = [prompt] if isinstance(prompt, str) else list(prompt)
    return [_prompt_text(item) for item in raw]


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Sequence):
        return "\n".join(str(line) for line in prompt)
    return str(prompt)


def _validate_numlines(numlines: int | Sequence[int], count: int) -> None:
    if isinstance(numlines, int):
        if numlines != 1:
            raise ValueError("inputdlg2 supports one editable line per prompt")
        return
    values = [int(value) for value in numlines]
    if len(values) != count or any(value != 1 for value in values):
        raise ValueError("inputdlg2 supports one editable line per prompt")
