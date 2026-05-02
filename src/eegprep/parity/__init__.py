"""Parity harness utilities for validating EEGPrep against EEGLAB."""

from .compare import (
    ComparisonResult,
    compare_eeg_struct,
    compare_events_epochs,
    compare_numeric,
    compare_stage_outputs,
    compare_visual_output,
    compare_workflow_trace,
)
from .config import (
    ParityCase,
    ParityDeviation,
    ParityManifest,
    ToleranceProfile,
    default_deviations_path,
    default_manifest_path,
    load_deviations,
    load_manifest,
)
from .oracle import (
    ArtifactOracle,
    MatlabBatchOracle,
    MatlabEngineOracle,
    OracleBackend,
    detect_oracle_backends,
    resolve_oracle_backend,
)
from .report import render_markdown_report, write_json_report, write_markdown_report

__all__ = [
    "ArtifactOracle",
    "ComparisonResult",
    "MatlabBatchOracle",
    "MatlabEngineOracle",
    "OracleBackend",
    "ParityCase",
    "ParityDeviation",
    "ParityManifest",
    "ToleranceProfile",
    "compare_eeg_struct",
    "compare_events_epochs",
    "compare_numeric",
    "compare_stage_outputs",
    "compare_visual_output",
    "compare_workflow_trace",
    "default_deviations_path",
    "default_manifest_path",
    "detect_oracle_backends",
    "load_deviations",
    "load_manifest",
    "render_markdown_report",
    "resolve_oracle_backend",
    "write_json_report",
    "write_markdown_report",
]
