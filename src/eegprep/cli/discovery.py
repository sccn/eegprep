"""Self-describing CLI capabilities, schemas, examples, and skill content."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from eegprep.cli.core import EEGPrepCLIError, ok


CLI_SKILL_NAME = "eegprep-cli"


def capabilities() -> dict[str, Any]:
    return ok(
        "eegprep.capabilities.v1",
        commands={
            "inspect": {
                "description": "Inspect EEGLAB dataset metadata, events, channels, or ICA fields.",
                "inputs": ["eeglab_set"],
                "outputs": ["json"],
                "supports_json": True,
                "supports_dry_run": False,
            },
            "validate": {
                "description": "Validate an EEG dataset and return stable warning/error codes.",
                "inputs": ["eeglab_set"],
                "outputs": ["json"],
                "supports_json": True,
                "supports_dry_run": False,
            },
            "resample": _write_capability("Resample a dataset to a new sampling rate."),
            "rereference": _write_capability("Apply average or channel-based rereferencing."),
            "filter": _write_capability("Apply high-pass, low-pass, band-pass, or notch FIR filtering."),
            "clean": _write_capability("Run clean_rawdata/ASR-style artifact cleaning."),
            "epoch": _write_capability("Extract epochs around event types."),
            "ica": _write_capability("Run ICA decomposition."),
            "pipeline": {
                "description": "Validate, plan, and run YAML preprocessing pipelines.",
                "inputs": ["eegprep_pipeline_yaml"],
                "outputs": ["eeglab_set", "manifest", "report"],
                "supports_json": True,
                "supports_dry_run": True,
            },
            "batch": {
                "description": "Run a pipeline sequentially across explicit input files or BIDS subjects.",
                "inputs": ["eeglab_set", "bids_eeg", "eegprep_pipeline_yaml"],
                "outputs": ["per_subject_results", "manifest"],
                "supports_json": True,
                "supports_dry_run": True,
            },
            "qc": {
                "description": "Compute dataset quality-control metrics and recommendations.",
                "inputs": ["eeglab_set"],
                "outputs": ["json", "html"],
                "supports_json": True,
                "supports_dry_run": False,
            },
            "report": {
                "description": "Generate an HTML dataset report with paired JSON-compatible QC metrics.",
                "inputs": ["eeglab_set"],
                "outputs": ["html", "manifest"],
                "supports_json": True,
                "supports_dry_run": False,
            },
            "software_info": {
                "description": "Report NumPy build backends, loaded thread-pool libraries, and CPU metadata.",
                "inputs": [],
                "outputs": ["text", "json"],
                "supports_json": True,
                "supports_dry_run": False,
            },
            "bids": {
                "description": "Validate, import, and export BIDS EEG datasets.",
                "inputs": ["bids_eeg", "eeglab_set"],
                "outputs": ["bids_eeg", "eeglab_set", "json"],
                "supports_json": True,
                "supports_dry_run": False,
            },
            "migrate": {
                "description": "Inspect EEGLAB history, compare datasets, and convert simple MATLAB histories.",
                "inputs": ["eeglab_set", "matlab_script"],
                "outputs": ["json", "eegprep_pipeline_yaml"],
                "supports_json": True,
                "supports_dry_run": False,
            },
            "skills": {
                "description": "List and retrieve bundled agent-facing CLI skill content.",
                "outputs": ["markdown", "json"],
                "supports_json": True,
                "supports_dry_run": False,
            },
        },
    )


def command_schema(command: str) -> dict[str, Any]:
    schemas = {
        "inspect": {
            "schema_version": "eegprep.schema.command.inspect.v1",
            "syntax": "eegprep inspect [dataset|events|channels|ica] <path> --json",
            "required": ["path"],
            "properties": {"kind": {"enum": ["dataset", "events", "channels", "ica"]}},
        },
        "pipeline": pipeline_schema()["schema"],
        "resample": _command_schema(
            "resample",
            required=["input", "freq"],
            properties={
                "freq": {"type": "number"},
                "engine": {"enum": ["poly", "scipy"], "default": "poly"},
            },
        ),
        "rereference": _command_schema(
            "rereference",
            required=["input", "method"],
            properties={
                "method": {"enum": ["average", "channels"], "default": "average"},
                "channels": {"type": "array", "items": {"type": "string"}},
                "exclude": {"type": "array", "items": {"type": "string"}},
                "keep_ref": {"type": "boolean", "default": False},
                "huber": {"type": "number"},
                "refica": {"enum": ["on", "off", "backwardcomp", "remove"], "default": "on"},
            },
        ),
        "filter": _command_schema(
            "filter",
            required=["input"],
            properties={
                "highpass": {"type": "number"},
                "lowpass": {"type": "number"},
                "notch": {"type": "number"},
                "notch_width": {"type": "number", "default": 2.0},
                "order": {"type": "integer"},
                "minphase": {"type": "boolean", "default": False},
                "usefftfilt": {"type": "boolean", "default": False},
            },
        ),
        "clean": _command_schema(
            "clean",
            required=["input", "method"],
            properties={
                "method": {"enum": ["asr", "rawdata"], "default": "asr"},
                "burst_criterion": {"type": "number", "default": 20.0},
                "burst_rejection": {"type": "boolean", "default": False},
                "distance": {"enum": ["euclidean", "riemannian"], "default": "euclidean"},
                "flatline_criterion": {"type": "number"},
                "channel_criterion": {"type": "number"},
                "line_noise_criterion": {"type": "number"},
                "window_criterion": {"type": "number"},
                "highpass": {"type": "array", "items": {"type": "number"}},
            },
        ),
        "epoch": _command_schema(
            "epoch",
            required=["input", "event_type", "tmin", "tmax"],
            properties={
                "event_type": {"type": "array", "items": {"type": "string"}},
                "tmin": {"type": "number"},
                "tmax": {"type": "number"},
                "new_name": {"type": "string"},
            },
        ),
        "ica": _command_schema(
            "ica",
            required=["input", "method"],
            properties={
                "method": {"enum": ["runica", "picard", "amica", "runamica15"], "default": "runica"},
                "seed": {"type": "integer"},
                "deterministic": {"type": "boolean", "default": True},
                "maxsteps": {"type": "integer"},
                "pca": {"type": "integer"},
                "extended": {"type": "integer"},
                "channels": {"type": "array", "items": {"type": "string"}},
                "option": {"type": "array", "items": {"type": "string"}},
                "reorder": {"type": "boolean", "default": True},
            },
        ),
        "batch": {
            "schema_version": "eegprep.schema.command.batch.v1",
            "syntax": "eegprep batch run <inputs...> --pipeline <config.yaml> --output-dir <dir> --json",
            "required": ["pipeline"],
            "properties": {
                "inputs": {"type": "array", "items": {"type": "string"}},
                "bids_root": {"type": "string"},
                "subjects": {"type": "array", "items": {"type": "string"}},
                "output_dir": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
        },
        "qc": {
            "schema_version": "eegprep.schema.command.qc.v1",
            "syntax": "eegprep qc <input.set> --json OR eegprep qc report <input.set> --html <report.html> --json",
            "required": ["input"],
            "properties": {
                "input": {"type": "string"},
                "html": {"type": "string"},
                "manifest": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
        },
        "report": {
            "schema_version": "eegprep.schema.command.report.v1",
            "syntax": "eegprep report <input.set> --output <report.html> --json",
            "required": ["input", "output"],
            "properties": {
                "input": {"type": "string"},
                "output": {"type": "string"},
                "manifest": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
        },
        "software_info": {
            "schema_version": "eegprep.schema.command.software_info.v1",
            "syntax": "eegprep software_info [--json]",
            "required": [],
            "properties": {},
        },
        "bids": {
            "schema_version": "eegprep.schema.command.bids.v1",
            "syntax": "eegprep bids <validate|import|export> ... --json",
            "required": ["subcommand"],
            "properties": {
                "root": {"type": "string"},
                "input": {"type": "string"},
                "output": {"type": "string"},
                "bids_root": {"type": "string"},
                "subject": {"type": "string"},
                "task": {"type": "string"},
            },
        },
        "migrate": {
            "schema_version": "eegprep.schema.command.migrate.v1",
            "syntax": "eegprep migrate <history|compare|convert-script> ... --json",
            "required": ["subcommand"],
            "properties": {
                "left": {"type": "string"},
                "right": {"type": "string"},
                "script": {"type": "string"},
                "output": {"type": "string"},
            },
        },
        "skills": {
            "schema_version": "eegprep.schema.command.skills.v1",
            "syntax": "eegprep skills <list|get|path> [name] --json",
            "required": ["subcommand"],
            "properties": {"name": {"type": "string"}, "full": {"type": "boolean", "default": False}},
        },
        "validate": {
            "schema_version": "eegprep.schema.command.validate.v1",
            "syntax": "eegprep validate <input.set> --json",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
        },
    }
    if command not in schemas:
        raise EEGPrepCLIError("COMMAND_NOT_IMPLEMENTED", f"No schema is available for command: {command}")
    return ok(f"eegprep.schema.command.{command}.v1", schema=schemas[command])


def pipeline_schema() -> dict[str, Any]:
    return ok(
        "eegprep.schema.pipeline.v1",
        schema={
            "type": "object",
            "required": ["schema_version", "input", "output", "steps"],
            "properties": {
                "schema_version": {"const": "eegprep.pipeline.v1"},
                "input": {"type": "object", "required": ["path"]},
                "output": {"type": "object", "required": ["directory"]},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {
                                "enum": ["filter", "rereference", "resample", "clean", "epoch", "ica", "qc", "report"]
                            }
                        },
                    },
                },
            },
        },
    )


def examples(name: str) -> dict[str, Any]:
    registry = {
        "inspect": [
            "eegprep inspect dataset sample_data/eeglab_data.set --json",
            "eegprep inspect events sample_data/eeglab_data.set --json",
        ],
        "pipeline": [
            "eegprep pipeline validate preprocessing.yaml --json",
            "eegprep pipeline plan preprocessing.yaml --json",
            "eegprep pipeline run preprocessing.yaml --json",
        ],
        "batch": [
            "eegprep batch run sub-01.set sub-02.set --pipeline preprocessing.yaml --output-dir derivatives/eegprep --json",
            "eegprep batch run --bids-root bids_root --subjects 01 02 --pipeline preprocessing.yaml --output-dir derivatives/eegprep --json",
        ],
        "filter": [
            "eegprep filter input.set --highpass 1 --lowpass 40 --output filtered.set --json",
        ],
        "resample": ["eegprep resample input.set --freq 128 --output resampled.set --json"],
        "rereference": ["eegprep rereference input.set --method average --output reref.set --json"],
        "clean": ["eegprep clean input.set --method asr --output clean.set --json"],
        "epoch": ["eegprep epoch input.set --event-type target --tmin -0.2 --tmax 0.8 --output epochs.set --json"],
        "ica": ["eegprep ica input.set --method runica --seed 42 --output ica.set --json"],
        "validate": ["eegprep validate sample_data/eeglab_data.set --json"],
        "qc": [
            "eegprep qc sample_data/eeglab_data.set --json",
            "eegprep qc report sample_data/eeglab_data.set --html qc.html --json",
        ],
        "report": ["eegprep report sample_data/eeglab_data.set --output report.html --json"],
        "software_info": ["eegprep software_info", "eegprep software_info --json"],
        "bids": [
            "eegprep bids validate bids_root --json",
            "eegprep bids export input.set --bids-root bids_out --subject 01 --task rest --json",
        ],
        "migrate": [
            "eegprep migrate history sample_data/eeglab_data.set --json",
            "eegprep migrate compare left.set right.set --json",
        ],
        "skills": ["eegprep skills list --json", "eegprep skills get eegprep-cli"],
    }
    if name not in registry:
        raise EEGPrepCLIError("COMMAND_NOT_IMPLEMENTED", f"No examples are available for: {name}")
    return ok("eegprep.examples.v1", name=name, examples=registry[name])


def skills_list() -> dict[str, Any]:
    return ok(
        "eegprep.skills.v1",
        skills=[
            {
                "name": CLI_SKILL_NAME,
                "description": "Core EEGPrep CLI usage guide for AI agents.",
                "path": str(_skill_path(CLI_SKILL_NAME)),
            }
        ],
    )


def skill_text(name: str, *, full: bool = False) -> str:
    path = _skill_path(name)
    if not path.is_file():
        raise EEGPrepCLIError("INPUT_FILE_NOT_FOUND", f"Unknown bundled skill: {name}")
    text = path.read_text(encoding="utf-8")
    if full:
        return text
    marker = "\n## Extended Reference\n"
    return text.split(marker, 1)[0].rstrip() + "\n" if marker in text else text


def skill_path(name: str) -> str:
    path = _skill_path(name)
    if not path.is_file():
        raise EEGPrepCLIError("INPUT_FILE_NOT_FOUND", f"Unknown bundled skill: {name}")
    return str(path)


def _skill_path(name: str) -> Any:
    return files("eegprep").joinpath("resources").joinpath("skills").joinpath(f"{name}.md")


def _write_capability(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "inputs": ["eeglab_set"],
        "outputs": ["eeglab_set", "manifest"],
        "supports_json": True,
        "supports_dry_run": False,
        "requires_output_or_overwrite": True,
    }


def _command_schema(command: str, *, required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": f"eegprep.schema.command.{command}.v1",
        "type": "object",
        "required": required,
        "anyOf": [{"required": ["output"]}, {"required": ["overwrite"]}],
        "properties": {
            "input": {"type": "string"},
            "output": {"type": "string"},
            "manifest": {"type": "string"},
            "overwrite": {"type": "boolean", "default": False},
            "json": {"type": "boolean", "default": False},
            **properties,
        },
    }
