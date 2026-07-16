"""Core EEG dataset transform commands for the headless CLI.

This module is dispatcher-neutral: the top-level CLI mounts these commands via
:func:`register_subcommands`, and a standalone ``python -m eegprep.cli.commands.transforms``
harness runs the same handlers for local testing.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

from eegprep.cli.core import (
    EEGPrepCLIError,
    MANIFEST_SCHEMA_VERSION as _MANIFEST_SCHEMA_VERSION,
    build_manifest,
    file_sha256,
    utc_now,
    write_manifest_file,
)
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_runica import pop_runica
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew


logger = logging.getLogger(__name__)

RESULT_SCHEMA_VERSION = "eegprep.transform_result.v1"
MANIFEST_SCHEMA_VERSION = _MANIFEST_SCHEMA_VERSION
DATASET_OUTPUT_TYPE = "eeglab_set"


@dataclass
class TransformResult:
    eeg: dict[str, Any]
    history: str
    parameters: dict[str, Any]
    deterministic: bool = True
    warnings: list[str] | None = None


class CliTransformError(EEGPrepCLIError):
    """Structured CLI transform failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: Path | str | None = None,
        suggestion: str | None = None,
        exit_code: int = 1,
    ) -> None:
        super().__init__(code, message, path=path, suggestion=suggestion, exit_code=exit_code)


def register_subcommands(subparsers: argparse._SubParsersAction, *, include_common_flags: bool = True) -> None:
    """Register transform commands on an argparse subparser collection."""

    _add_resample_parser(subparsers, include_common_flags=include_common_flags)
    _add_rereference_parser(subparsers, include_common_flags=include_common_flags)
    _add_filter_parser(subparsers, include_common_flags=include_common_flags)
    _add_clean_parser(subparsers, include_common_flags=include_common_flags)
    _add_epoch_parser(subparsers, include_common_flags=include_common_flags)
    _add_ica_parser(subparsers, include_common_flags=include_common_flags)


def build_parser() -> argparse.ArgumentParser:
    """Build a standalone parser for this module's local test harness."""

    parser = argparse.ArgumentParser(prog="python -m eegprep.cli.commands.transforms")
    subparsers = parser.add_subparsers(dest="transform_command", required=True)
    register_subcommands(subparsers)
    return parser


def run_transform_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run a parsed transform command and return its JSON-serializable result."""

    _configure_logging(args)
    with redirect_stdout(sys.stderr):
        return _run_transform_command(args)


def main(argv: list[str] | None = None) -> int:
    """Route module execution through the canonical top-level CLI dispatcher."""
    from eegprep.cli.main import main as cli_main

    return cli_main(sys.argv[1:] if argv is None else argv)


def _run_transform_command(args: argparse.Namespace) -> dict[str, Any]:
    input_path = _existing_input_path(Path(args.input))
    output_path = _resolve_output_path(input_path, args)
    manifest_path = _resolve_manifest_path(output_path, args)
    _validate_output_targets(input_path, output_path, manifest_path, overwrite=bool(args.overwrite))

    started_at = utc_now()
    logger.info("Loading %s", input_path)
    eeg = pop_loadset(str(input_path))
    input_files = _dataset_file_records(input_path, eeg)

    logger.info("Running %s", args.transform_command)
    result = apply_loaded_transform(eeg, args)
    if result.history:
        _append_history(result.eeg, result.history)

    logger.info("Saving %s", output_path)
    saved_eeg = pop_saveset(result.eeg, str(output_path))
    finished_at = utc_now()
    output_files = _dataset_file_records(output_path, saved_eeg)

    manifest = build_manifest(
        command=args.transform_command,
        input_files=input_files,
        output_files=output_files,
        parameters=result.parameters,
        history=result.history,
        deterministic=result.deterministic,
        warnings=result.warnings or [],
        started_at=started_at,
        finished_at=finished_at,
    )
    write_manifest_file(manifest_path, manifest, overwrite=True)

    return {
        "status": "ok",
        "schema_version": RESULT_SCHEMA_VERSION,
        "command": args.transform_command,
        "output": {"path": str(output_path), "type": DATASET_OUTPUT_TYPE},
        "manifest": {"path": str(manifest_path), "data": manifest},
        "history": result.history,
        "summary": _dataset_summary(saved_eeg),
        "warnings": result.warnings or [],
    }


def apply_transform(eeg: dict[str, Any], command: str, parameters: Mapping[str, Any] | None = None) -> TransformResult:
    """Apply a transform to an already-loaded EEG dict using CLI-equivalent defaults."""
    return apply_loaded_transform(eeg, transform_args(command, parameters or {}))


def apply_loaded_transform(eeg: dict[str, Any], args: argparse.Namespace) -> TransformResult:
    """Apply a parsed transform command to an already-loaded EEG dict."""
    command = args.transform_command
    if command == "resample":
        return _resample(eeg, args)
    if command == "rereference":
        return _rereference(eeg, args)
    if command == "filter":
        return _filter(eeg, args)
    if command == "clean":
        return _clean(eeg, args)
    if command == "epoch":
        return _epoch(eeg, args)
    if command == "ica":
        return _ica(eeg, args)
    raise CliTransformError("COMMAND_NOT_IMPLEMENTED", f"Transform command is not implemented: {command}")


def transform_args(command: str, parameters: Mapping[str, Any]) -> argparse.Namespace:
    """Return an argparse namespace matching the direct transform subcommand defaults."""
    normalized = _normalize_transform_command(command)
    params = dict(parameters)
    common: dict[str, Any] = {
        "transform_command": normalized,
        "input": "",
        "output": None,
        "manifest": None,
        "overwrite": False,
        "json": False,
        "quiet": False,
        "verbose": False,
        "no_progress": False,
    }
    if normalized == "resample":
        return argparse.Namespace(**common, freq=params.get("freq"), engine=params.get("engine") or "poly")
    if normalized == "rereference":
        return argparse.Namespace(
            **common,
            method=str(params.get("method") or "average").lower(),
            channels=_list_or_none(params.get("channels", params.get("ref"))),
            exclude=_list_or_none(params.get("exclude")),
            keep_ref=bool(params.get("keep_ref", params.get("keepref", False))),
            huber=params.get("huber"),
            refica=params.get("refica") or "on",
        )
    if normalized == "filter":
        return argparse.Namespace(
            **common,
            highpass=params.get("highpass"),
            lowpass=params.get("lowpass"),
            notch=params.get("notch"),
            notch_width=params.get("notch_width", 2.0),
            order=params.get("order"),
            minphase=bool(params.get("minphase", False)),
            usefftfilt=bool(params.get("usefftfilt", False)),
        )
    if normalized == "clean":
        return argparse.Namespace(
            **common,
            method=str(params.get("method") or "asr").lower(),
            burst_criterion=params.get("burst_criterion", 20.0),
            burst_rejection=bool(params.get("burst_rejection", False)),
            distance=str(params.get("distance") or "euclidean").lower(),
            flatline_criterion=params.get("flatline_criterion"),
            channel_criterion=params.get("channel_criterion"),
            line_noise_criterion=params.get("line_noise_criterion"),
            window_criterion=params.get("window_criterion"),
            highpass=params.get("highpass"),
        )
    if normalized == "epoch":
        return argparse.Namespace(
            **common,
            event_type=_list_or_empty(params.get("event_type", params.get("event_types"))),
            tmin=params.get("tmin"),
            tmax=params.get("tmax"),
            new_name=params.get("new_name", params.get("newname")),
        )
    if normalized == "ica":
        return argparse.Namespace(
            **common,
            method=str(params.get("method") or "runica").lower(),
            seed=params.get("seed"),
            deterministic=bool(params.get("deterministic", True)),
            maxsteps=params.get("maxsteps"),
            pca=params.get("pca"),
            extended=params.get("extended"),
            channels=_list_or_none(params.get("channels")),
            option=_ica_option_list(params),
            reorder=bool(params.get("reorder", True)),
        )
    raise CliTransformError("COMMAND_NOT_IMPLEMENTED", f"Transform command is not implemented: {command}")


def _resample(eeg: dict[str, Any], args: argparse.Namespace) -> TransformResult:
    output, history = pop_resample(eeg, args.freq, engine=args.engine, gui=False, return_com=True)
    return TransformResult(
        eeg=output,
        history=history,
        parameters={"freq": float(args.freq), "engine": args.engine or "poly"},
    )


def _rereference(eeg: dict[str, Any], args: argparse.Namespace) -> TransformResult:
    if args.method == "average":
        ref: list[Any] = []
    else:
        if not args.channels:
            raise CliTransformError(
                "CONFIG_SCHEMA_ERROR",
                "rereference --method channels requires --channels.",
                suggestion="Pass channel labels or EEGLAB-facing 1-based channel indices.",
            )
        ref = channel_tokens(args.channels, numeric_base=1, output_base=0)

    kwargs: dict[str, Any] = {"refica": args.refica}
    if args.exclude:
        kwargs["exclude"] = channel_tokens(args.exclude, numeric_base=1, output_base=0)
    if args.keep_ref:
        kwargs["keepref"] = "on"
    if args.huber is not None:
        kwargs["huber"] = float(args.huber)

    output, history = pop_reref(eeg, ref, gui=False, return_com=True, **kwargs)
    return TransformResult(
        eeg=output,
        history=history,
        parameters={
            "method": args.method,
            "channels": args.channels or [],
            "exclude": args.exclude or [],
            "keep_ref": bool(args.keep_ref),
            "huber": args.huber,
            "refica": args.refica,
        },
    )


def _filter(eeg: dict[str, Any], args: argparse.Namespace) -> TransformResult:
    if args.highpass is None and args.lowpass is None and args.notch is None:
        raise CliTransformError(
            "CONFIG_SCHEMA_ERROR",
            "filter requires at least one of --highpass, --lowpass, or --notch.",
        )

    output = eeg
    history_parts: list[str] = []
    if args.highpass is not None or args.lowpass is not None:
        output, history = pop_eegfiltnew(
            output,
            locutoff=args.highpass,
            hicutoff=args.lowpass,
            filtorder=args.order,
            plotfreqz=False,
            minphase=bool(args.minphase),
            usefftfilt=bool(args.usefftfilt),
            gui=False,
            return_com=True,
        )
        if history:
            history_parts.append(history)

    if args.notch is not None:
        notch_width = float(args.notch_width)
        lower_edge = float(args.notch) - notch_width / 2.0
        upper_edge = float(args.notch) + notch_width / 2.0
        if lower_edge <= 0:
            raise CliTransformError("CONFIG_SCHEMA_ERROR", "notch minus half notch_width must be positive.")
        output, history = pop_eegfiltnew(
            output,
            locutoff=lower_edge,
            hicutoff=upper_edge,
            filtorder=args.order,
            revfilt=True,
            plotfreqz=False,
            minphase=bool(args.minphase),
            usefftfilt=bool(args.usefftfilt),
            gui=False,
            return_com=True,
        )
        if history:
            history_parts.append(history)

    return TransformResult(
        eeg=output,
        history="\n".join(history_parts),
        parameters={
            "highpass": args.highpass,
            "lowpass": args.lowpass,
            "notch": args.notch,
            "notch_width": args.notch_width,
            "order": args.order,
            "minphase": bool(args.minphase),
            "usefftfilt": bool(args.usefftfilt),
        },
    )


def _clean(eeg: dict[str, Any], args: argparse.Namespace) -> TransformResult:
    options = _clean_options(args)
    output, history = pop_clean_rawdata(eeg, gui=False, return_com=True, **options)
    return TransformResult(
        eeg=output,
        history=history,
        parameters={
            "method": args.method,
            "burst_criterion": args.burst_criterion,
            "burst_rejection": bool(args.burst_rejection),
            "distance": args.distance,
            "flatline_criterion": args.flatline_criterion,
            "channel_criterion": args.channel_criterion,
            "line_noise_criterion": args.line_noise_criterion,
            "window_criterion": args.window_criterion,
            "highpass": args.highpass,
        },
    )


def _epoch(eeg: dict[str, Any], args: argparse.Namespace) -> TransformResult:
    types = [_parse_scalar(value) for value in args.event_type] if args.event_type else []
    options: dict[str, Any] = {"epochinfo": "yes"}
    if args.new_name:
        options["newname"] = args.new_name
    output, history = pop_epoch(
        eeg,
        types,
        [float(args.tmin), float(args.tmax)],
        gui=False,
        return_com=True,
        **options,
    )
    return TransformResult(
        eeg=output,
        history=history,
        parameters={
            "event_type": args.event_type or [],
            "tmin": float(args.tmin),
            "tmax": float(args.tmax),
            "new_name": args.new_name,
        },
    )


def _ica(eeg: dict[str, Any], args: argparse.Namespace) -> TransformResult:
    options = _ica_options(args)
    chanind = channel_tokens(args.channels, numeric_base=1, output_base=1) if args.channels else None
    method = "runamica15" if args.method == "amica" else args.method
    warnings = []
    if args.seed is not None and method != "runica":
        warnings.append(f"--seed is passed to the {method} backend only if it supports a seed option.")
    output, history = pop_runica(
        eeg,
        icatype=method,
        options=options,
        chanind=chanind,
        reorder="on" if args.reorder else "off",
        gui=False,
        return_com=True,
    )
    return TransformResult(
        eeg=output,
        history=history,
        parameters={
            "method": args.method,
            "seed": args.seed,
            "deterministic": bool(args.deterministic),
            "maxsteps": args.maxsteps,
            "pca": args.pca,
            "extended": args.extended,
            "channels": args.channels or [],
            "reorder": bool(args.reorder),
            "options": args.option or [],
        },
        deterministic=_ica_is_deterministic(method, options, args),
        warnings=warnings,
    )


def _clean_options(args: argparse.Namespace) -> dict[str, Any]:
    distance = "Riemannian" if args.distance == "riemannian" else "Euclidean"
    highpass = _clean_highpass(args.highpass)
    if args.method == "asr":
        return {
            "FlatlineCriterion": _optional_float_or_off(args.flatline_criterion, default="off"),
            "ChannelCriterion": _optional_float_or_off(args.channel_criterion, default="off"),
            "LineNoiseCriterion": _optional_float_or_off(args.line_noise_criterion, default="off"),
            "Highpass": highpass,
            "BurstCriterion": float(args.burst_criterion),
            "BurstRejection": bool(args.burst_rejection),
            "WindowCriterion": _optional_float_or_off(args.window_criterion, default="off"),
            "Distance": distance,
        }

    options: dict[str, Any] = {"Distance": distance, "BurstRejection": bool(args.burst_rejection)}
    if args.flatline_criterion is not None:
        options["FlatlineCriterion"] = float(args.flatline_criterion)
    if args.channel_criterion is not None:
        options["ChannelCriterion"] = float(args.channel_criterion)
    if args.line_noise_criterion is not None:
        options["LineNoiseCriterion"] = float(args.line_noise_criterion)
    if highpass != "off":
        options["Highpass"] = highpass
    if args.burst_criterion is not None:
        options["BurstCriterion"] = float(args.burst_criterion)
    if args.window_criterion is not None:
        options["WindowCriterion"] = float(args.window_criterion)
    return options


def _clean_highpass(value: Any) -> list[float] | str:
    if value in (None, "", "off"):
        return "off"
    if isinstance(value, str):
        values: Any = [item for item in value.replace(",", " ").split() if item]
    else:
        values = value
    if not isinstance(values, list | tuple) or len(values) != 2:
        raise CliTransformError(
            "CONFIG_SCHEMA_ERROR",
            "clean highpass must contain [low high] transition band values.",
            path="highpass",
            suggestion="Use --highpass LOW HIGH or highpass: [LOW, HIGH] in pipeline config.",
        )
    try:
        low, high = (float(values[0]), float(values[1]))
    except (TypeError, ValueError) as exc:
        raise CliTransformError(
            "CONFIG_SCHEMA_ERROR",
            "clean highpass values must be numeric.",
            path="highpass",
        ) from exc
    if low <= 0 or high <= 0 or low >= high:
        raise CliTransformError(
            "CONFIG_SCHEMA_ERROR",
            "clean highpass must be positive and ordered as [low, high].",
            path="highpass",
        )
    return [low, high]


def _ica_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if args.seed is not None:
        options["seed"] = int(args.seed)
    elif args.deterministic and args.method == "runica":
        options["rndreset"] = "off"
    if args.maxsteps is not None:
        options["maxsteps"] = int(args.maxsteps)
    if args.pca is not None:
        options["pca"] = int(args.pca)
    if args.extended is not None:
        options["extended"] = int(args.extended)
    for item in args.option or []:
        if "=" not in item:
            raise CliTransformError(
                "CONFIG_SCHEMA_ERROR",
                "--option values must use KEY=VALUE syntax.",
                suggestion="Example: --option lrate=0.001",
            )
        key, value = item.split("=", 1)
        options[key.strip()] = _parse_scalar(value)
    return options


def _normalize_transform_command(command: str) -> str:
    normalized = str(command).strip().lower()
    aliases = {"reref": "rereference", "re_reference": "rereference"}
    return aliases.get(normalized, normalized)


def _list_or_none(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _list_or_empty(value: Any) -> list[Any]:
    return _list_or_none(value) or []


def _ica_option_list(parameters: Mapping[str, Any]) -> list[str]:
    options: list[str] = []
    raw_options = parameters.get("options", parameters.get("option", []))
    if isinstance(raw_options, Mapping):
        options.extend(f"{key}={value}" for key, value in raw_options.items())
    elif isinstance(raw_options, str):
        options.append(raw_options)
    else:
        options.extend(str(item) for item in raw_options or [])
    for key in ("lrate",):
        if key in parameters:
            options.append(f"{key}={parameters[key]}")
    return options


def _ica_is_deterministic(method: str, options: dict[str, Any], args: argparse.Namespace) -> bool:
    if method != "runica":
        return args.seed is not None
    if "seed" in options:
        return True
    if "rndreset" not in options:
        return bool(args.deterministic)
    return str(options["rndreset"]).lower() in {"off", "false", "0", "no"}


def _add_common_arguments(parser: argparse.ArgumentParser, *, include_common_flags: bool) -> None:
    parser.add_argument("input", help="Input EEGLAB .set file.")
    parser.add_argument("--output", help="Output EEGLAB .set file.")
    parser.add_argument("--manifest", help="Manifest JSON path. Defaults to <output>.manifest.json.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing over an existing output.")
    if include_common_flags:
        parser.add_argument("--json", action="store_true", help="Write clean JSON to stdout.")
        parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose progress logging.")
        parser.add_argument("--no-progress", action="store_true", help="Disable progress logging.")


def _set_transform_defaults(parser: argparse.ArgumentParser, command: str) -> None:
    parser.set_defaults(transform_command=command, handler=run_transform_command)


def _add_resample_parser(subparsers: argparse._SubParsersAction, *, include_common_flags: bool) -> None:
    parser = subparsers.add_parser("resample", help="Resample an EEGLAB dataset.")
    _add_common_arguments(parser, include_common_flags=include_common_flags)
    parser.add_argument("--freq", type=float, required=True, help="New sampling rate in Hz.")
    parser.add_argument("--engine", choices=("poly", "scipy"), default="poly", help="Resampling engine.")
    _set_transform_defaults(parser, "resample")


def _add_rereference_parser(subparsers: argparse._SubParsersAction, *, include_common_flags: bool) -> None:
    parser = subparsers.add_parser("rereference", aliases=("reref",), help="Re-reference an EEGLAB dataset.")
    _add_common_arguments(parser, include_common_flags=include_common_flags)
    parser.add_argument("--method", choices=("average", "channels"), default="average")
    parser.add_argument("--channels", nargs="+", help="Reference channel labels or EEGLAB-facing 1-based indices.")
    parser.add_argument("--exclude", nargs="+", help="Channels to exclude from re-referencing.")
    parser.add_argument("--keep-ref", action="store_true", help="Keep reference channels in the output.")
    parser.add_argument("--huber", type=float, help="Huber average reference threshold in microvolts.")
    parser.add_argument("--refica", choices=("on", "off", "backwardcomp", "remove"), default="on")
    _set_transform_defaults(parser, "rereference")


def _add_filter_parser(subparsers: argparse._SubParsersAction, *, include_common_flags: bool) -> None:
    parser = subparsers.add_parser("filter", help="Filter an EEGLAB dataset.")
    _add_common_arguments(parser, include_common_flags=include_common_flags)
    parser.add_argument("--highpass", type=float, help="Lower pass-band edge in Hz.")
    parser.add_argument("--lowpass", type=float, help="Upper pass-band edge in Hz.")
    parser.add_argument("--notch", type=float, help="Notch center frequency in Hz.")
    parser.add_argument("--notch-width", type=float, default=2.0, help="Notch width in Hz.")
    parser.add_argument("--order", type=int, help="FIR filter order. Defaults to the pop_eegfiltnew heuristic.")
    parser.add_argument("--minphase", action="store_true", help="Use minimum-phase causal FIR filtering.")
    parser.add_argument("--usefftfilt", action="store_true", help="Use frequency-domain FIR filtering.")
    _set_transform_defaults(parser, "filter")


def _add_clean_parser(subparsers: argparse._SubParsersAction, *, include_common_flags: bool) -> None:
    parser = subparsers.add_parser("clean", help="Clean continuous data with clean_rawdata.")
    _add_common_arguments(parser, include_common_flags=include_common_flags)
    parser.add_argument("--method", choices=("asr", "rawdata"), default="asr")
    parser.add_argument("--burst-criterion", type=float, default=20.0)
    parser.add_argument("--burst-rejection", action="store_true")
    parser.add_argument("--distance", choices=("euclidean", "riemannian"), default="euclidean")
    parser.add_argument("--flatline-criterion", type=float)
    parser.add_argument("--channel-criterion", type=float)
    parser.add_argument("--line-noise-criterion", type=float)
    parser.add_argument("--window-criterion", type=float)
    parser.add_argument("--highpass", nargs=2, type=float, metavar=("LOW", "HIGH"))
    _set_transform_defaults(parser, "clean")


def _add_epoch_parser(subparsers: argparse._SubParsersAction, *, include_common_flags: bool) -> None:
    parser = subparsers.add_parser("epoch", help="Extract event-locked epochs.")
    _add_common_arguments(parser, include_common_flags=include_common_flags)
    parser.add_argument("--event-type", action="append", default=[], help="Event type to epoch. Repeat for several.")
    parser.add_argument("--tmin", type=float, required=True, help="Epoch start in seconds.")
    parser.add_argument("--tmax", type=float, required=True, help="Epoch end in seconds.")
    parser.add_argument("--new-name", help="Output dataset setname.")
    _set_transform_defaults(parser, "epoch")


def _add_ica_parser(subparsers: argparse._SubParsersAction, *, include_common_flags: bool) -> None:
    parser = subparsers.add_parser("ica", help="Run ICA decomposition.")
    _add_common_arguments(parser, include_common_flags=include_common_flags)
    parser.add_argument("--method", choices=("runica", "picard", "amica", "runamica15"), default="runica")
    parser.add_argument("--seed", type=int, help="Random seed for backends that support one.")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--maxsteps", type=int, help="Maximum ICA training steps.")
    parser.add_argument("--pca", type=int, help="Number of principal components to retain.")
    parser.add_argument("--extended", type=int, help="runica extended-ICA option.")
    parser.add_argument("--channels", nargs="+", help="Channel labels or EEGLAB-facing 1-based indices for ICA.")
    parser.add_argument("--option", action="append", help="Backend option as KEY=VALUE. Repeat for several.")
    parser.add_argument("--reorder", dest="reorder", action="store_true", default=True)
    parser.add_argument("--no-reorder", dest="reorder", action="store_false")
    _set_transform_defaults(parser, "ica")


def _existing_input_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.exists():
        raise CliTransformError(
            "INPUT_FILE_NOT_FOUND",
            f"Input file does not exist: {expanded}",
            path=expanded,
            suggestion="Check the dataset path before running the transform.",
        )
    if not expanded.is_file():
        raise CliTransformError("UNSUPPORTED_FORMAT", f"Input path is not a file: {expanded}", path=expanded)
    if expanded.suffix.lower() not in {".set", ".mat"}:
        raise CliTransformError(
            "UNSUPPORTED_FORMAT",
            f"Unsupported input format: {expanded.suffix or '<none>'}",
            path=expanded,
            suggestion="Use an EEGLAB .set or MATLAB .mat dataset file.",
        )
    return expanded.resolve()


def _resolve_output_path(input_path: Path, args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    if args.overwrite:
        return input_path
    raise CliTransformError(
        "OUTPUT_REQUIRED",
        "Transform commands require --output unless --overwrite is explicit.",
        suggestion="Pass --output <path> or pass --overwrite to replace the input dataset.",
    )


def _resolve_manifest_path(output_path: Path, args: argparse.Namespace) -> Path:
    if args.manifest:
        return Path(args.manifest).expanduser().resolve()
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def _validate_output_targets(input_path: Path, output_path: Path, manifest_path: Path, *, overwrite: bool) -> None:
    if output_path == manifest_path:
        raise CliTransformError(
            "OUTPUT_PATH_COLLISION",
            f"Output and manifest paths must be different: {output_path}",
            path=output_path,
            suggestion="Choose a separate --manifest path or omit --manifest to use the default sidecar path.",
        )
    if output_path.exists() and not overwrite:
        raise CliTransformError(
            "OUTPUT_EXISTS",
            f"Output already exists: {output_path}",
            path=output_path,
            suggestion="Choose a new --output path or pass --overwrite.",
        )
    if manifest_path.exists() and not overwrite:
        raise CliTransformError(
            "OUTPUT_EXISTS",
            f"Manifest already exists: {manifest_path}",
            path=manifest_path,
            suggestion="Choose a new --manifest path or pass --overwrite.",
        )
    if _same_path(input_path, output_path) and not overwrite:
        raise CliTransformError(
            "OUTPUT_EXISTS",
            f"Output path matches input path: {output_path}",
            path=output_path,
            suggestion="Pass --overwrite to replace the input dataset.",
        )


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return left == right


def _dataset_file_records(set_path: Path, eeg: dict[str, Any]) -> list[dict[str, Any]]:
    paths = [set_path]
    datfile = _string_value(eeg.get("datfile"))
    if datfile:
        dat_path = Path(datfile)
        if not dat_path.is_absolute():
            dat_path = set_path.parent / dat_path.name
        if dat_path.exists():
            paths.append(dat_path.resolve())
    return [_file_record(path) for path in paths if path.exists()]


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "type": _file_type(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".set":
        return "eeglab_set"
    if suffix == ".fdt":
        return "eeglab_fdt"
    return "file"


def _dataset_summary(eeg: dict[str, Any]) -> dict[str, Any]:
    data = eeg.get("data")
    shape = list(np.asarray(data).shape) if data is not None and not isinstance(data, str) else []
    return {
        "setname": _string_value(eeg.get("setname")),
        "nbchan": int(eeg.get("nbchan", shape[0] if shape else 0) or 0),
        "pnts": int(eeg.get("pnts", shape[1] if len(shape) > 1 else 0) or 0),
        "trials": int(eeg.get("trials", shape[2] if len(shape) > 2 else 1) or 0),
        "srate": float(eeg.get("srate", 0) or 0),
        "data_shape": shape,
    }


def _append_history(eeg: dict[str, Any], command: str) -> None:
    existing = _string_value(eeg.get("history")).rstrip()
    command = command.strip()
    if not command:
        return
    eeg["history"] = f"{existing}\n{command}\n" if existing else f"{command}\n"


def channel_tokens(
    values: list[Any] | tuple[Any, ...],
    *,
    numeric_base: int,
    output_base: int,
    path: str | None = None,
) -> list[Any]:
    """Return parsed channel labels or converted EEGLAB-facing indices."""
    tokens: list[Any] = []
    for value in values:
        scalar = _parse_scalar(value)
        if isinstance(scalar, int):
            if scalar < numeric_base:
                raise CliTransformError(
                    "CONFIG_SCHEMA_ERROR",
                    "Channel indices are EEGLAB-facing 1-based values.",
                    path=path,
                    suggestion="Use channel index 1 for the first channel.",
                )
            tokens.append(scalar - numeric_base + output_base)
        else:
            tokens.append(scalar)
    return tokens


def _parse_scalar(value: Any) -> Any:
    text = str(value).strip()
    lower = text.lower()
    if lower in {"on", "off", "true", "false", "yes", "no"}:
        return text
    try:
        integer = int(text)
    except ValueError:
        pass
    else:
        return integer
    try:
        return float(text)
    except ValueError:
        return text


def _optional_float_or_off(value: float | None, *, default: str) -> float | str:
    return default if value is None else float(value)


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            return _string_value(value.reshape(-1)[0])
    return str(value)


def _configure_logging(args: argparse.Namespace) -> None:
    if getattr(args, "quiet", False) or getattr(args, "no_progress", False):
        level = logging.WARNING
    elif getattr(args, "verbose", False):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr, force=True)


if __name__ == "__main__":
    raise SystemExit(main())
