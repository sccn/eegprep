"""Agent-friendly EEGPrep pipeline command support."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import yaml

from eegprep.cli.commands.qc import compute_qc_metrics
from eegprep.cli.core import (
    EEGPrepCLIError as CommandError,
    build_manifest,
    command_error as error_result,
    command_ok as success_result,
    emit_command_result as emit_result,
    file_sha256,
    utc_now,
    write_json_file,
    write_manifest_file,
)
from eegprep.cli.dataset import load_dataset as load_eeg_dataset
from eegprep.cli.reporting import write_report_html
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_runica import pop_runica
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew


PIPELINE_SCHEMA_VERSION = "eegprep.pipeline.v1"
SUPPORTED_STEP_NAMES = {
    "filter",
    "rereference",
    "resample",
    "clean",
    "epoch",
    "ica",
    "qc",
    "report",
}
MUTATING_STEP_NAMES = {"filter", "rereference", "resample", "clean", "epoch", "ica"}
STEP_ALIASES = {
    "reref": "rereference",
    "rereference": "rereference",
    "re_reference": "rereference",
}


def validate_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    """Validate a YAML pipeline config without mutating inputs or outputs."""
    try:
        config, warnings = _load_normalized_config(config_path)
        return success_result(
            "pipeline validate",
            config_path=str(config_path),
            validation={
                "valid": True,
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "warnings": warnings,
                "normalized_steps": config["steps"],
            },
        )
    except CommandError as exc:
        return error_result("pipeline validate", exc)


def plan_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    """Plan a YAML pipeline config without loading or writing EEG data."""
    try:
        config, warnings = _load_normalized_config(config_path)
        plan = _build_plan(config, warnings=warnings)
        return success_result("pipeline plan", config_path=str(config_path), plan=plan)
    except CommandError as exc:
        return error_result("pipeline plan", exc)


def run_pipeline_config(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    manifest_path: str | Path | None = None,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Run a YAML pipeline config, or return the plan when ``dry_run`` is true."""
    started_at = utc_now()
    try:
        config, warnings = _load_normalized_config(config_path)
        if manifest_path is not None:
            config["output"]["manifest_path"] = str(_resolve_path(manifest_path, Path(config_path).parent))
        if overwrite is not None:
            config["output"]["overwrite"] = bool(overwrite)
        plan = _build_plan(config, warnings=warnings)
        if dry_run:
            return success_result("pipeline run", config_path=str(config_path), dry_run=True, plan=plan)

        _preflight_writes(plan["output_files"], overwrite=config["output"]["overwrite"])
        EEG = load_eeg_dataset(config["input"]["path"])
        histories: list[str] = []
        output_files: list[dict[str, Any]] = []
        latest_qc: dict[str, Any] | None = None

        for index, step in enumerate(config["steps"]):
            try:
                EEG, history, step_outputs, latest_qc = _run_step(
                    EEG,
                    step,
                    config=config,
                    latest_qc=latest_qc,
                    overwrite=config["output"]["overwrite"],
                )
            except CommandError:
                raise
            except Exception as exc:
                raise CommandError(
                    "PIPELINE_STEP_FAILED",
                    f"Pipeline step {index + 1} ({step['name']}) failed: {exc}",
                    path=f"steps[{index}]",
                    suggestion="Inspect the step parameters and run pipeline plan before retrying.",
                    details={"step": step},
                ) from exc
            if history:
                histories.append(history)
            output_files.extend(step_outputs)

        dataset_output = config["output"].get("dataset_path")
        if dataset_output and _has_mutating_steps(config["steps"]):
            output_files.extend(_save_dataset(EEG, dataset_output, history="\n".join(histories)))

        manifest_path = config["output"]["manifest_path"]
        finished_at = utc_now()
        manifest = build_manifest(
            command="pipeline run",
            input_files=[{"path": config["input"]["path"], "sha256": file_sha256(config["input"]["path"])}],
            output_files=output_files,
            parameters=_public_config(config),
            history="\n".join(histories),
            started_at=started_at,
            finished_at=finished_at,
            deterministic=_is_deterministic(config),
            warnings=plan["warnings"],
        )
        manifest_entry = write_manifest_file(manifest_path, manifest, overwrite=config["output"]["overwrite"])
        output_files.append(manifest_entry)
        return success_result(
            "pipeline run",
            config_path=str(config_path),
            dry_run=False,
            output_files=output_files,
            manifest=str(manifest_path),
            history=histories,
            qc=latest_qc,
        )
    except CommandError as exc:
        return error_result("pipeline run", exc)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register ``pipeline`` commands with an argparse dispatcher."""
    parser = subparsers.add_parser("pipeline", help="Validate, plan, or run YAML EEGPrep pipelines.")
    pipeline_subparsers = parser.add_subparsers(dest="pipeline_action", required=True, parser_class=type(parser))

    validate_parser = pipeline_subparsers.add_parser("validate", help="Validate a pipeline YAML file.")
    validate_parser.add_argument("config", help="Pipeline YAML config")
    validate_parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    validate_parser.set_defaults(handler=handle_registered)

    plan_parser = pipeline_subparsers.add_parser("plan", help="Plan a pipeline YAML file without writes.")
    plan_parser.add_argument("config", help="Pipeline YAML config")
    plan_parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    plan_parser.set_defaults(handler=handle_registered)

    run_parser = pipeline_subparsers.add_parser("run", help="Run a pipeline YAML file.")
    run_parser.add_argument("config", help="Pipeline YAML config")
    run_parser.add_argument("--dry-run", action="store_true", help="Return the plan without loading or writing data")
    run_parser.add_argument("--manifest", help="Override the default pipeline manifest path")
    run_parser.add_argument("--overwrite", action="store_true", help="Overwrite configured outputs")
    run_parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    run_parser.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    run_parser.add_argument("--verbose", action="store_true", help="Enable verbose progress logging")
    run_parser.add_argument("--no-progress", action="store_true", help="Disable progress logging")
    run_parser.set_defaults(handler=handle_registered)
    return parser


def handle_registered(args: argparse.Namespace) -> dict[str, Any]:
    """Handle ``pipeline`` when mounted into a larger dispatcher."""
    if args.pipeline_action == "validate":
        return validate_pipeline_config(args.config)
    if args.pipeline_action == "plan":
        return plan_pipeline_config(args.config)
    if args.pipeline_action == "run":
        _configure_logging(args)
        with redirect_stdout(sys.stderr):
            return run_pipeline_config(
                args.config,
                dry_run=args.dry_run,
                manifest_path=args.manifest,
                overwrite=True if args.overwrite else None,
            )
    raise CommandError("COMMAND_NOT_IMPLEMENTED", f"Unknown pipeline action: {args.pipeline_action}")


def main(argv: list[str] | None = None) -> int:
    """Standalone module entry point for local command testing."""
    parser = argparse.ArgumentParser(prog="eegprep pipeline")
    subparsers = parser.add_subparsers(dest="pipeline_action", required=True)
    register(subparsers)
    args = parser.parse_args(["pipeline", *(sys.argv[1:] if argv is None else argv)])
    result = handle_registered(args)
    return emit_result(result, json_output=True)


def _load_normalized_config(config_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(config_path)
    if not path.exists():
        raise CommandError(
            "INPUT_FILE_NOT_FOUND",
            f"Pipeline config not found: {path}",
            path=str(path),
            suggestion="Check the YAML config path and retry.",
        )
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise CommandError(
            "UNSUPPORTED_FORMAT",
            f"Unsupported pipeline config format: {path.suffix or '<none>'}",
            path=str(path),
            suggestion="Use a .yaml or .yml pipeline config.",
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CommandError(
            "CONFIG_SCHEMA_ERROR",
            f"Pipeline YAML could not be parsed: {exc}",
            path=str(path),
            suggestion="Fix the YAML syntax and rerun pipeline validate.",
        ) from exc
    return _normalize_config(raw, config_path=path)


def _normalize_config(raw: Any, *, config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    base_dir = config_path.parent
    if not isinstance(raw, dict):
        _add_error(errors, "", "Pipeline config must be a mapping.")
        _raise_schema_errors(errors, config_path)

    schema_version = raw.get("schema_version")
    if schema_version != PIPELINE_SCHEMA_VERSION:
        _add_error(
            errors,
            "schema_version",
            f"schema_version must be {PIPELINE_SCHEMA_VERSION!r}.",
        )

    input_config = raw.get("input")
    if not isinstance(input_config, dict):
        _add_error(errors, "input", "input must be a mapping with a path.")
        input_config = {}
    input_path_value = input_config.get("path")
    if not isinstance(input_path_value, str) or not input_path_value.strip():
        _add_error(errors, "input.path", "input.path is required.")
        input_path = base_dir / "<missing-input>"
    else:
        input_path = _resolve_path(input_path_value, base_dir)
        if not input_path.exists():
            _add_error(errors, "input.path", f"Input dataset does not exist: {input_path}")
        elif input_path.suffix.lower() not in {".set", ".mat"}:
            _add_error(errors, "input.path", "Only EEGLAB .set/.mat inputs are supported by pipeline run.")

    output_config = raw.get("output")
    if not isinstance(output_config, dict):
        _add_error(errors, "output", "output must be a mapping with a directory.")
        output_config = {}
    output_dir_value = output_config.get("directory")
    if not isinstance(output_dir_value, str) or not output_dir_value.strip():
        _add_error(errors, "output.directory", "output.directory is required.")
        output_dir = base_dir / "<missing-output>"
    else:
        output_dir = _resolve_path(output_dir_value, base_dir)

    steps_raw = raw.get("steps")
    steps: list[dict[str, Any]] = []
    if not isinstance(steps_raw, list) or not steps_raw:
        _add_error(errors, "steps", "steps must be a non-empty list.")
    else:
        for index, step_raw in enumerate(steps_raw):
            steps.append(_normalize_step(step_raw, index=index, errors=errors, warnings=warnings))

    output = _normalize_output(
        output_config, output_dir=output_dir, input_path=input_path, steps=steps, base_dir=base_dir
    )
    warnings.extend(_dependency_warnings(steps))
    if errors:
        _raise_schema_errors(errors, config_path)
    return (
        {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "input": {
                "path": str(input_path),
                "format": str(input_config.get("format") or "eeglab"),
            },
            "output": output,
            "steps": steps,
        },
        warnings,
    )


def _normalize_output(
    raw: dict[str, Any],
    *,
    output_dir: Path,
    input_path: Path,
    steps: list[dict[str, Any]],
    base_dir: Path,
) -> dict[str, Any]:
    dataset_path = _output_path(raw.get("dataset"), output_dir / f"{input_path.stem}_eegprep.set", base_dir)
    manifest_path = _output_path(raw.get("manifest"), output_dir / "eegprep_manifest.json", base_dir)
    qc_path = _output_path(raw.get("qc"), output_dir / "qc.json", base_dir) if _has_step(steps, "qc") else None
    report_path = (
        _output_path(raw.get("report"), output_dir / "report.html", base_dir) if _has_step(steps, "report") else None
    )
    return {
        "directory": str(output_dir),
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "qc_path": str(qc_path) if qc_path else None,
        "report_path": str(report_path) if report_path else None,
        "overwrite": bool(raw.get("overwrite", False)),
    }


def _normalize_step(
    raw: Any,
    *,
    index: int,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    path = f"steps[{index}]"
    if not isinstance(raw, dict):
        _add_error(errors, path, "Each step must be a mapping.")
        return {"name": "<invalid>", "parameters": {}}
    name = _normalize_step_name(raw.get("name"))
    if name not in SUPPORTED_STEP_NAMES:
        _add_error(errors, f"{path}.name", f"Unsupported pipeline step: {raw.get('name')!r}.")
        return {"name": name or "<missing>", "parameters": {}}
    parameters = {key: value for key, value in raw.items() if key != "name"}
    _validate_step_parameters(name, parameters, path=path, errors=errors, warnings=warnings)
    return {"name": name, "parameters": parameters}


def _validate_step_parameters(
    name: str,
    parameters: dict[str, Any],
    *,
    path: str,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    if name == "resample":
        _positive_number(parameters.get("freq"), f"{path}.freq", errors, required=True)
    elif name == "filter":
        for key in ("highpass", "lowpass", "notch", "notch_width"):
            if key in parameters and parameters[key] is not None:
                _positive_number(parameters[key], f"{path}.{key}", errors)
        if not any(parameters.get(key) is not None for key in ("highpass", "lowpass", "notch")):
            _add_error(errors, path, "filter requires highpass, lowpass, or notch.")
        if parameters.get("notch") is not None and parameters.get("notch_width") is None:
            warnings.append(
                _warning(
                    "FILTER_NOTCH_WIDTH_DEFAULTED",
                    f"{path}.notch",
                    "notch_width omitted; pipeline run will use a 2 Hz notch stop band.",
                )
            )
    elif name == "rereference":
        method = str(parameters.get("method") or "average").lower()
        if method not in {"average", "channels"}:
            _add_error(errors, f"{path}.method", "rereference method must be 'average' or 'channels'.")
        if method == "channels" and not (parameters.get("channels") or parameters.get("ref")):
            _add_error(errors, f"{path}.channels", "channels rereference requires channels or ref.")
    elif name == "clean":
        method = str(parameters.get("method") or "asr").lower()
        if method != "asr":
            _add_error(errors, f"{path}.method", "clean currently supports method='asr'.")
        for key in ("burst_criterion", "channel_criterion", "line_noise_criterion", "window_criterion"):
            if key in parameters and parameters[key] != "off":
                _positive_number(parameters[key], f"{path}.{key}", errors)
    elif name == "epoch":
        if not (parameters.get("event_type") or parameters.get("event_types")):
            _add_error(errors, f"{path}.event_type", "epoch requires event_type or event_types.")
        tmin = _number(parameters.get("tmin"), f"{path}.tmin", errors, required=True)
        tmax = _number(parameters.get("tmax"), f"{path}.tmax", errors, required=True)
        if tmin is not None and tmax is not None and tmin >= tmax:
            _add_error(errors, f"{path}.tmax", "epoch tmax must be greater than tmin.")
    elif name == "ica":
        method = str(parameters.get("method") or "runica").lower()
        if method not in {"runica", "picard"}:
            _add_error(errors, f"{path}.method", "ica currently supports method='runica' or method='picard'.")
    elif name == "report":
        fmt = str(parameters.get("format") or "html").lower()
        if fmt != "html":
            _add_error(errors, f"{path}.format", "report currently supports format='html'.")


def _build_plan(config: dict[str, Any], *, warnings: list[dict[str, Any]]) -> dict[str, Any]:
    output_files = _planned_output_files(config)
    overwrite = config["output"]["overwrite"]
    destructive_actions = [
        {"path": item["path"], "action": "overwrite"}
        for item in output_files
        if Path(item["path"]).exists() and overwrite
    ]
    blocked_writes = [
        {"path": item["path"], "code": "OUTPUT_EXISTS"}
        for item in output_files
        if Path(item["path"]).exists() and not overwrite
    ]
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "input": config["input"],
        "normalized_steps": config["steps"],
        "output_files": output_files,
        "estimated_destructive_actions": destructive_actions,
        "blocked_writes": blocked_writes,
        "warnings": warnings,
        "mutates_input": False,
    }


def _planned_output_files(config: dict[str, Any]) -> list[dict[str, str]]:
    output = config["output"]
    files: list[dict[str, str]] = []
    if _has_mutating_steps(config["steps"]):
        files.append({"path": output["dataset_path"], "type": "eeglab_set"})
    if output.get("qc_path"):
        files.append({"path": output["qc_path"], "type": "json"})
    if output.get("report_path"):
        files.append({"path": output["report_path"], "type": "html_report"})
    files.append({"path": output["manifest_path"], "type": "manifest"})
    return files


def _preflight_writes(output_files: list[dict[str, str]], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [item for item in output_files if Path(item["path"]).exists()]
    if existing:
        raise CommandError(
            "OUTPUT_EXISTS",
            f"Output already exists: {existing[0]['path']}",
            path=existing[0]["path"],
            suggestion="Use overwrite: true in the config, pass --overwrite, or choose a different output path.",
            details={"existing_outputs": existing},
        )


def _run_step(
    EEG: dict[str, Any],
    step: dict[str, Any],
    *,
    config: dict[str, Any],
    latest_qc: dict[str, Any] | None,
    overwrite: bool,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, Any] | None]:
    name = step["name"]
    parameters = step["parameters"]
    if name == "filter":
        EEG, history = _apply_filter(EEG, parameters)
        return EEG, history, [], latest_qc
    if name == "rereference":
        EEG, history = _apply_rereference(EEG, parameters)
        return EEG, history, [], latest_qc
    if name == "resample":
        EEG, history = _apply_resample(EEG, parameters)
        return EEG, history, [], latest_qc
    if name == "clean":
        EEG, history = _apply_clean(EEG, parameters)
        return EEG, history, [], latest_qc
    if name == "epoch":
        EEG, history = _apply_epoch(EEG, parameters)
        return EEG, history, [], latest_qc
    if name == "ica":
        EEG, history = _apply_ica(EEG, parameters)
        return EEG, history, [], latest_qc
    if name == "qc":
        qc_payload = compute_qc_metrics(EEG, dataset_path=config["input"]["path"])
        entry = write_json_file(config["output"]["qc_path"], qc_payload, overwrite=overwrite)
        return EEG, "", [entry], qc_payload
    if name == "report":
        metrics = latest_qc or compute_qc_metrics(EEG, dataset_path=config["input"]["path"])
        entry = write_report_html(
            EEG,
            config["output"]["report_path"],
            metrics=metrics,
            dataset_path=config["input"]["path"],
            title="EEGPrep Pipeline Report",
            overwrite=overwrite,
        )
        return EEG, "", [entry], metrics
    raise CommandError("COMMAND_NOT_IMPLEMENTED", f"Pipeline step is not implemented: {name}")


def _apply_filter(EEG: dict[str, Any], parameters: dict[str, Any]) -> tuple[dict[str, Any], str]:
    histories = []
    highpass = parameters.get("highpass")
    lowpass = parameters.get("lowpass")
    if highpass is not None or lowpass is not None:
        EEG, history = pop_eegfiltnew(
            EEG,
            locutoff=highpass,
            hicutoff=lowpass,
            plotfreqz=False,
            gui=False,
            return_com=True,
        )
        histories.append(history)
    notch = parameters.get("notch")
    if notch is not None:
        width = float(parameters.get("notch_width") or 2.0)
        lower_edge = float(notch) - width / 2
        if lower_edge <= 0:
            raise CommandError(
                "CONFIG_SCHEMA_ERROR",
                "notch minus half notch_width must be positive.",
                path="steps[].notch",
                suggestion="Increase notch or decrease notch_width so the notch stop band stays above 0 Hz.",
            )
        EEG, history = pop_eegfiltnew(
            EEG,
            locutoff=lower_edge,
            hicutoff=float(notch) + width / 2,
            revfilt=True,
            plotfreqz=False,
            gui=False,
            return_com=True,
        )
        histories.append(history)
    return EEG, "\n".join(item for item in histories if item)


def _apply_rereference(EEG: dict[str, Any], parameters: dict[str, Any]) -> tuple[dict[str, Any], str]:
    method = str(parameters.get("method") or "average").lower()
    ref = [] if method == "average" else _channel_indices(parameters.get("channels", parameters.get("ref")))
    return pop_reref(EEG, ref, gui=False, return_com=True)


def _apply_resample(EEG: dict[str, Any], parameters: dict[str, Any]) -> tuple[dict[str, Any], str]:
    return pop_resample(EEG, float(parameters["freq"]), gui=False, return_com=True)


def _apply_clean(EEG: dict[str, Any], parameters: dict[str, Any]) -> tuple[dict[str, Any], str]:
    clean_kwargs: dict[str, Any] = {}
    key_map = {
        "burst_criterion": "BurstCriterion",
        "channel_criterion": "ChannelCriterion",
        "line_noise_criterion": "LineNoiseCriterion",
        "window_criterion": "WindowCriterion",
        "flatline_criterion": "FlatlineCriterion",
        "highpass": "Highpass",
    }
    for source, target in key_map.items():
        if source in parameters:
            clean_kwargs[target] = parameters[source]
    return pop_clean_rawdata(EEG, gui=False, return_com=True, **clean_kwargs)


def _apply_epoch(EEG: dict[str, Any], parameters: dict[str, Any]) -> tuple[dict[str, Any], str]:
    event_types = parameters.get("event_types", parameters.get("event_type"))
    limits = [float(parameters["tmin"]), float(parameters["tmax"])]
    return pop_epoch(EEG, event_types, limits, gui=False, return_com=True)


def _apply_ica(EEG: dict[str, Any], parameters: dict[str, Any]) -> tuple[dict[str, Any], str]:
    method = str(parameters.get("method") or "runica").lower()
    options = {}
    for key in ("seed", "maxsteps", "extended", "lrate"):
        if key in parameters:
            options[key] = parameters[key]
    return pop_runica(EEG, icatype=method, options=options or None, gui=False, return_com=True)


def _save_dataset(EEG: dict[str, Any], output_path: str | Path, *, history: str) -> list[dict[str, Any]]:
    if history:
        existing_history = str(EEG.get("history") or "")
        EEG["history"] = (existing_history + "\n" + history).strip()
    saved = pop_saveset(EEG, str(output_path))
    target = Path(saved.get("filepath", Path(output_path).parent)) / str(saved.get("filename", Path(output_path).name))
    outputs = [{"path": str(target), "type": "eeglab_set", "sha256": file_sha256(target)}]
    datfile = saved.get("datfile")
    if datfile:
        dat_path = Path(saved.get("filepath", target.parent)) / str(datfile)
        if dat_path.exists():
            outputs.append({"path": str(dat_path), "type": "eeglab_fdt", "sha256": file_sha256(dat_path)})
    return outputs


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "input": config["input"],
        "output": {
            "directory": config["output"]["directory"],
            "overwrite": config["output"]["overwrite"],
        },
        "steps": config["steps"],
    }


def _dependency_warnings(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings = []
    for index, step in enumerate(steps):
        if step["name"] == "ica" and str(step["parameters"].get("method") or "runica").lower() == "picard":
            if importlib.util.find_spec("picard") is None:
                warnings.append(
                    _warning(
                        "DEPENDENCY_MISSING",
                        f"steps[{index}].method",
                        "python-picard is required for ICA method 'picard'.",
                    )
                )
    return warnings


def _is_deterministic(config: dict[str, Any]) -> bool:
    for step in config["steps"]:
        if step["name"] == "ica" and "seed" not in step["parameters"]:
            return False
        if step["name"] == "clean":
            return False
    return True


def _has_mutating_steps(steps: list[dict[str, Any]]) -> bool:
    return any(step["name"] in MUTATING_STEP_NAMES for step in steps)


def _has_step(steps: list[dict[str, Any]], name: str) -> bool:
    return any(step["name"] == name for step in steps)


def _normalize_step_name(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_")
    return STEP_ALIASES.get(name, name)


def _output_path(value: Any, default: Path, base_dir: Path) -> Path:
    if value is None or value == "":
        return default.resolve()
    return _resolve_path(str(value), base_dir)


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _channel_indices(value: Any) -> list[Any]:
    values = value if isinstance(value, list | tuple) else [value]
    refs = []
    for item in values:
        if isinstance(item, int | float) and float(item).is_integer():
            if int(item) < 1:
                raise CommandError(
                    "CONFIG_SCHEMA_ERROR",
                    "Pipeline channel indices are EEGLAB-facing 1-based values.",
                    path="steps[].channels",
                    suggestion="Use channel index 1 for the first channel.",
                )
            refs.append(int(item) - 1)
            continue
        text = str(item)
        if text.isdecimal():
            number = int(text)
            if number < 1:
                raise CommandError(
                    "CONFIG_SCHEMA_ERROR",
                    "Pipeline channel indices are EEGLAB-facing 1-based values.",
                    path="steps[].channels",
                    suggestion="Use channel index 1 for the first channel.",
                )
            refs.append(number - 1)
        else:
            refs.append(item)
    return refs


def _number(value: Any, path: str, errors: list[dict[str, Any]], *, required: bool = False) -> float | None:
    if value is None:
        if required:
            _add_error(errors, path, "Value is required.")
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        _add_error(errors, path, "Value must be numeric.")
        return None


def _positive_number(
    value: Any,
    path: str,
    errors: list[dict[str, Any]],
    *,
    required: bool = False,
) -> float | None:
    number = _number(value, path, errors, required=required)
    if number is not None and number <= 0:
        _add_error(errors, path, "Value must be positive.")
    return number


def _add_error(errors: list[dict[str, Any]], path: str, message: str) -> None:
    errors.append({"path": path, "message": message})


def _raise_schema_errors(errors: list[dict[str, Any]], config_path: Path) -> None:
    raise CommandError(
        "CONFIG_SCHEMA_ERROR",
        f"Pipeline config is invalid ({len(errors)} error(s)).",
        path=str(config_path),
        suggestion="Run pipeline validate and fix the reported config paths.",
        details={"errors": errors},
    )


def _warning(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


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
