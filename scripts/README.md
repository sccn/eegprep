# Scripts

This directory contains maintainer commands and manual workflows that are not
part of the installed `eegprep` Python package.

## Maintained Commands

- `make_release.py`: release helper for building and publishing EEGPrep.

## Incubating Workflows

The remaining scripts are important manual or exploratory workflows, especially
for whole-BIDS-dataset processing and MATLAB/EEGLAB parity checks. They are kept
here because they exercise larger workflows than the unit tests, which usually
focus on one file or one function.

Do not treat these scripts as polished public examples yet. Many still contain
machine-specific paths, temporary assumptions, or parity/debugging code.

When one of these workflows is cleaned up enough to be user-facing, move it into
the Sphinx examples under `docs/source/examples/` with runnable instructions and
remove the old script from this directory.

