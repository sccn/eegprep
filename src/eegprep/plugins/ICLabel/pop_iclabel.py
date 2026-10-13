"""EEGLAB-style pop wrapper for ICLabel."""

from __future__ import annotations

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.ICLabel.iclabel import iclabel, iclabel_async, require_sync_backend


_VERSIONS = ("default", "lite", "beta")


def pop_iclabel(
    EEG,
    icversion: str | None = None,
    *,
    gui: bool | None = None,
    renderer=None,
    engine=None,
    return_com: bool = False,
):
    """Classify independent components with ICLabel, EEGLAB style.

    Args:
        EEG: EEG dataset, or a list of datasets, each with an ICA
            decomposition. ``None`` returns ``None`` (``(None, "")`` with
            ``return_com=True``) without doing anything.
        icversion: ICLabel network: ``'default'``, ``'lite'``, or ``'beta'``
            (case-insensitive). Only ``'default'`` ships with EEGPrep.
        gui: Show the version dialog first. Defaults to ``True`` when
            ``icversion`` is not given.
        renderer: Dialog renderer passed to ``inputgui``; ``None`` uses the
            default Qt renderer.
        engine: ``None`` for the packaged network, or ``'matlab'`` or
            ``'octave'``; see :func:`eegprep.plugins.ICLabel.iclabel.iclabel`.
        return_com: Also return the history command.

    Returns:
        The labeled dataset (or list of datasets); with ``return_com=True``, a
        ``(EEG, command)`` tuple whose command is
        ``"EEG = pop_iclabel(EEG, 'default');"`` for the default network.
        If the dialog is canceled, ``EEG`` is returned unchanged with an
        empty command.

    Raises:
        RuntimeError: Under Emscripten (Pyodide); use :func:`pop_iclabel_async`.
        ValueError: If a dataset has no ICA decomposition or ``icversion`` is
            not supported.
        NotImplementedError: If ``icversion`` is ``'lite'`` or ``'beta'`` with
            ``engine=None``.
        ImportError: If ``engine=None`` and onnxruntime is not installed; the
            message names the ``eegprep[iclabel]`` extra.
    """
    require_sync_backend(engine)
    icversion = _requested_icversion(EEG, icversion, gui, renderer)
    if icversion is None:
        return (EEG, "") if return_com else EEG
    output = [
        None if dataset is None else iclabel(dataset, algorithm=icversion, engine=engine)
        for dataset in _datasets_with_ica(EEG)
    ]
    output = output if isinstance(EEG, list) else output[0]
    return (output, _history_command(icversion)) if return_com else output


async def pop_iclabel_async(
    EEG,
    icversion: str | None = None,
    *,
    gui: bool | None = None,
    renderer=None,
    engine=None,
    return_com: bool = False,
):
    """Classify independent components with ICLabel from asynchronous code.

    The awaitable counterpart of :func:`pop_iclabel` and the ICLabel entry
    point under Emscripten (Pyodide), where the network runs through the
    host's ONNX Runtime Web session. It returns the same datasets as
    :func:`pop_iclabel`; with ``return_com=True`` only the history command
    differs, because it contains ``await``. See
    :func:`eegprep.plugins.ICLabel.iclabel.iclabel_async` for event-loop
    behavior.

    Args:
        EEG: EEG dataset, or a list of datasets, each with an ICA
            decomposition. ``None`` returns ``None`` (``(None, "")`` with
            ``return_com=True``) without doing anything.
        icversion: ICLabel network: ``'default'``, ``'lite'``, or ``'beta'``
            (case-insensitive). Only ``'default'`` ships with EEGPrep.
        gui: Show the version dialog first. Defaults to ``True`` when
            ``icversion`` is not given.
        renderer: Dialog renderer passed to ``inputgui``.
        engine: ``None`` for the packaged network, or ``'matlab'`` or
            ``'octave'`` on native platforms.
        return_com: Also return the history command.

    Returns:
        The labeled dataset (or list of datasets); with ``return_com=True``, a
        ``(EEG, command)`` tuple whose command is
        ``"EEG = await pop_iclabel_async(EEG, 'default');"`` for the default
        network, which replays in ``eegprep-console``. If the dialog is
        canceled, ``EEG`` is returned unchanged with an empty command.

    Raises:
        RuntimeError: For ``engine='matlab'`` or ``'octave'`` under Emscripten.
        ValueError: If a dataset has no ICA decomposition or ``icversion`` is
            not supported.
        NotImplementedError: If ``icversion`` is ``'lite'`` or ``'beta'`` with
            ``engine=None``.
        ImportError: On native platforms, if ``engine=None`` and onnxruntime is
            not installed; the message names the ``eegprep[iclabel]`` extra.
    """
    icversion = _requested_icversion(EEG, icversion, gui, renderer)
    if icversion is None:
        return (EEG, "") if return_com else EEG
    output = [
        None if dataset is None else await iclabel_async(dataset, algorithm=icversion, engine=engine)
        for dataset in _datasets_with_ica(EEG)
    ]
    output = output if isinstance(EEG, list) else output[0]
    return (output, _history_command(icversion, asynchronous=True)) if return_com else output


def _requested_icversion(EEG, icversion, gui, renderer):
    """Return the ICLabel version to run, or None when there is no dataset or the dialog was canceled."""
    if EEG is None:
        return None
    if gui is None:
        gui = icversion is None
    if gui:
        result = _run_gui(renderer=renderer)
        if result is None:
            return None
        icversion = result["icversion"]
    icversion = "default" if icversion is None else str(icversion).lower()
    if icversion not in _VERSIONS:
        raise ValueError("icversion must be one of 'default', 'lite', or 'beta'")
    return icversion


def _datasets_with_ica(EEG):
    """Return ``EEG`` as a list after checking each dataset's ICA weights; empty (``None``) slots pass through."""
    datasets = EEG if isinstance(EEG, list) else [EEG]
    for dataset in datasets:
        if dataset is not None:
            _require_ica(dataset)
    return datasets


def pop_iclabel_dialog_spec() -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_iclabel``."""
    return DialogSpec(
        title="ICLabel",
        function_name="pop_iclabel",
        eeglab_source="plugins/ICLabel/pop_iclabel.m",
        geometry=((1,), (1,)),
        size=(356, 199),
        show_help_button=False,
        controls=(
            ControlSpec("text", "Select which icversion of ICLabel to use:"),
            ControlSpec("popupmenu", "Default (recommended)|Lite|Beta", tag="icversion", value=1),
        ),
    )


def _run_gui(renderer=None):
    result = inputgui(pop_iclabel_dialog_spec(), renderer=renderer)
    if result is None:
        return None
    index = int(result.get("icversion", 1)) - 1
    index = max(0, min(index, len(_VERSIONS) - 1))
    return {"icversion": _VERSIONS[index]}


def _require_ica(EEG):
    weights = EEG.get("icaweights")
    if weights is None or np.asarray(weights).size == 0:
        raise ValueError("ICLabel requires an ICA decomposition. Run pop_runica first.")


def _history_command(icversion, *, asynchronous=False):
    if asynchronous:
        return f"EEG = await pop_iclabel_async(EEG, '{icversion}');"
    return f"EEG = pop_iclabel(EEG, '{icversion}');"
