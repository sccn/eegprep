# EEGPrep CLI Agent Guide

Use `eegprep` when an agent needs a headless, reproducible EEG workflow. Use
`eegprep-gui` and `eegprep-console` for human interactive work.

## Agent Rules

- Prefer `--json` for every command that returns data.
- Treat stdout as the command result. Logs, progress, and warnings belong on
  stderr.
- Run `eegprep capabilities --json` before guessing available commands.
- Run `eegprep schema command <name> --json` or
  `eegprep schema pipeline --json` before generating configs.
- Use `--dry-run`, `pipeline plan`, or `pipeline run --dry-run` before expensive
  or destructive work.
- Always write a new output path unless the user explicitly requests
  `--overwrite`.
- Read manifest JSON files after commands that write artifacts.
- Handle errors by stable `code` values such as `INPUT_FILE_NOT_FOUND`,
  `OUTPUT_EXISTS`, `CONFIG_SCHEMA_ERROR`, `ICA_NOT_FOUND`, and
  `BIDS_VALIDATION_FAILED`.

## Common Commands

```bash
eegprep inspect dataset sample_data/eeglab_data.set --json
eegprep validate sample_data/eeglab_data.set --json
eegprep resample sample_data/eeglab_data.set --freq 128 --output out.set --json
eegprep rereference input.set --method average --output reref.set --json
eegprep filter input.set --highpass 1 --lowpass 40 --output filtered.set --json
eegprep epoch input.set --event-type target --tmin -0.2 --tmax 0.8 --output epochs.set --json
eegprep pipeline validate preprocess.yaml --json
eegprep pipeline plan preprocess.yaml --json
eegprep pipeline run preprocess.yaml --json
eegprep batch run sub-01.set sub-02.set --pipeline preprocess.yaml --output-dir derivatives/eegprep --json
eegprep qc input.set --json
eegprep report input.set --output report.html --json
```

## Pipeline Skeleton

```yaml
schema_version: eegprep.pipeline.v1
input:
  path: sub-01.set
  format: eeglab
output:
  directory: derivatives/eegprep/sub-01
  overwrite: false
steps:
  - name: filter
    highpass: 1
    lowpass: 40
  - name: rereference
    method: average
  - name: resample
    freq: 128
  - name: qc
  - name: report
    format: html
```

## Extended Reference

EEGPrep's CLI is workflow-native, not module-native. Commands map to how EEG
researchers work: inspect, validate, filter, rereference, resample, epoch, run
ICA, compute QC, write reports, and run pipeline configs.

When generating commands:

- Use absolute or user-provided paths; do not assume the current directory
  unless the user gave it.
- Keep syntax boring: `eegprep <command> <input> [options] --output <path>
  --json`.
- Do not pipe JSON through commands that may write logs to stdout. EEGPrep keeps
  JSON clean on stdout; stderr may contain progress.
- Prefer config-driven `pipeline` for multi-step preprocessing because the YAML
  can be validated, planned, reviewed, and rerun.
- Preserve EEGLAB-facing terms in user text: `EEG`, `ALLEEG`, `CURRENTSET`,
  events, channel locations, ICA fields, and `pop_*` history.

## Recovery Patterns

- `INPUT_FILE_NOT_FOUND`: ask for a corrected path or inspect the parent
  directory.
- `OUTPUT_EXISTS`: choose a new output path unless the user explicitly approves
  `--overwrite`.
- `CONFIG_SCHEMA_ERROR`: run the matching schema command and fix the generated
  config.
- `ICA_NOT_FOUND`: run `eegprep inspect ica <file> --json` and only run
  component workflows after ICA fields exist.
- `BIDS_VALIDATION_FAILED`: run `eegprep bids validate <root> --json` and fix
  the reported BIDS structure first.
