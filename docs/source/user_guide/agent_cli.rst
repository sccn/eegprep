.. _agent_cli:

==========================
Agent-Friendly EEGPrep CLI
==========================

EEGPrep includes a headless ``eegprep`` command for agents, batch jobs, and
reproducible preprocessing pipelines. It complements the human-facing
``eegprep-gui`` and ``eegprep-console`` entry points; it does not replace the
shared GUI plus console workspace.

The CLI is organized around researcher workflows rather than internal modules:
inspect, validate, filter, rereference, resample, epoch, ICA, QC, reports,
pipelines, BIDS, and EEGLAB migration helpers.

Agent Contract
==============

Use these rules when automating EEGPrep:

* Use ``--json`` for machine-readable output.
* Treat stdout as the result stream. Progress, warnings, and logs go to stderr.
* Use stable error codes for recovery, such as ``INPUT_FILE_NOT_FOUND``,
  ``OUTPUT_EXISTS``, ``CONFIG_SCHEMA_ERROR``, ``ICA_NOT_FOUND``, and
  ``BIDS_VALIDATION_FAILED``.
* Prefer non-destructive outputs. Commands require ``--output`` unless
  ``--overwrite`` is explicit.
* Inspect plans and schemas before running expensive or mutating workflows.
* Read manifest files for provenance after commands that write artifacts.

The root help includes an agent start section:

.. code-block:: bash

   eegprep --help
   eegprep skills get eegprep-cli

Discovery Commands
==================

.. code-block:: bash

   eegprep capabilities --json
   eegprep schema command filter --json
   eegprep schema pipeline --json
   eegprep examples pipeline --json
   eegprep skills list --json
   eegprep skills get eegprep-cli

These commands let agents discover supported operations and version-matched
usage guidance without scraping docs.

Dataset Inspection And Validation
=================================

.. code-block:: bash

   eegprep inspect dataset sample_data/eeglab_data.set --json
   eegprep inspect events sample_data/eeglab_data.set --json
   eegprep inspect channels sample_data/eeglab_data.set --json
   eegprep inspect ica sample_data/eeglab_data.set --json
   eegprep validate sample_data/eeglab_data.set --json

Transform Commands
==================

Transform commands load an EEGLAB ``.set`` dataset, call the existing EEGPrep
``pop_*`` implementation, save an output dataset, and write a manifest.

.. code-block:: bash

   eegprep resample input.set --freq 128 --output resampled.set --json
   eegprep rereference input.set --method average --output reref.set --json
   eegprep filter input.set --highpass 1 --lowpass 40 --output filtered.set --json
   eegprep epoch input.set --event-type target --tmin -0.2 --tmax 0.8 --output epochs.set --json
   eegprep ica input.set --method runica --maxsteps 64 --output ica.set --json

Pipeline Configs
================

Use YAML pipelines for multi-step preprocessing because they can be validated,
planned, reviewed, and rerun.

.. code-block:: yaml

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

.. code-block:: bash

   eegprep pipeline validate preprocess.yaml --json
   eegprep pipeline plan preprocess.yaml --json
   eegprep pipeline run preprocess.yaml --dry-run --json
   eegprep pipeline run preprocess.yaml --json
   eegprep batch run sub-01.set sub-02.set --pipeline preprocess.yaml --output-dir derivatives/eegprep --json

QC And Reports
==============

.. code-block:: bash

   eegprep qc input.set --json
   eegprep qc report input.set --html qc.html --manifest qc.manifest.json --json
   eegprep report input.set --output report.html --manifest report.manifest.json --json

QC results include stable recommendation codes that an agent can reason over.
HTML reports are for human review; the paired JSON and manifests are for
automation.

BIDS And Migration
==================

.. code-block:: bash

   eegprep bids validate bids_root --json
   eegprep bids import bids_root --subject 01 --task rest --output sub-01.set --json
   eegprep bids export clean.set --bids-root bids_out --subject 01 --task rest --json
   eegprep migrate history old_pipeline.set --json
   eegprep migrate compare matlab_output.set eegprep_output.set --json
   eegprep migrate convert-script old_pipeline.m --output preprocess.yaml --json

Migration helpers can inspect EEGLAB command histories and compare datasets
without making normal EEGPrep CLI usage depend on MATLAB or an EEGLAB checkout.
Script conversion is intentionally best-effort and reports unsupported commands
instead of silently inventing behavior.
