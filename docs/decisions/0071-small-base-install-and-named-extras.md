# 0071. The next major version installs a small core and named extras

- Status: Proposed
- Date: 2026-11-03
- Supersedes: the "default install stays whole" clause of [ADR 0069](https://github.com/nemarOrg/nemar-cli/blob/main/.context/decisions/0069-the-browser-runtime-is-its-own-package-and-eegprep-stays-whole.md) (accepted 2026-09-20, owner Seyed Yahya Shirazi, in nemar-cli). That clause keeps every current dependency on `pip install eegprep`.
- Leaves in force: the rest of ADR 0069. `eegprep-lean` remains its own distribution. An extra can only add packages, so it cannot shrink the base install. ADR 0070, the browser recipe, is unchanged.
- Number: 0071, so this file does not collide with ADR 0070. It lives in this repository because `.context/` is gitignored here.

Suraj chose this shape, the same one pandas and MNE use: a small required install, named extras for the rest, one extra that restores today's install, and an error that names the extra to install. Arno and Seyed still have the questions at the end. Until those are answered, this document is a proposal and `pip install eegprep` does not change.

Measurements below are from the feasibility report on `develop` at `f22111a5` (pull request #517), walked from `uv.lock` for linux x86_64, Python 3.12. Sizes are compressed wheel bytes, which is what pip downloads.

## Decision

In the next major version, `pip install eegprep` installs numpy and scipy, plus the tiny pure-Python helpers core code still imports at startup (`pyyaml`, `packaging`, `threadpoolctl`, and `colorama` only if logging still needs it). That install includes the lazy public names, `eegprep.help()`, and the NEMAR reader once its source lives in this tree.

Everything else is an extra:

| Extra | Installs | What it is for |
| --- | --- | --- |
| `io` | h5py, eeglabio, pyedflib | Reading and writing EEG files, including EDF and BDF |
| `plot` | matplotlib | Topography and other figures |
| `bids` | pybids | BIDS import and export |
| `ica` | python-picard | Picard ICA |
| `mne` | mne | Conversion to and from MNE |
| `eeglab` | oct2py | The Octave engine |
| `sys` | psutil | RAM reservations for ASR and job planning |
| `iclabel` | ONNX runtime and the model file | ICLabel. `netICL.mat` and the dipfit MRI leave the base wheel with this extra |
| `gui` | unchanged | Qt dialogs |
| `console` | unchanged | `eegprep-console` |
| `torch` | unchanged | Torch models |
| `docs` | unchanged | The documentation build |
| `full` | the union of the extras above | Today's install, base plus the extras that exist today |

`eegprep[full]` is the upgrade path. A script that today runs after `pip install eegprep` runs after `pip install eegprep[full]`.

Calling a function whose library is missing raises `ImportError`, not a bare `ModuleNotFoundError`. The message names the library, the function or feature, the command `pip install "eegprep[<extra>]"`, and `pip install "eegprep[full]"`. `help()` prints the same extra next to each public name, read from source, without importing the module.

## eegprep-lean

`eegprep-lean` stays a second wheel until OSA moves off the pin. The wheel is built from one shared copy of the reader source, mapped to the import name `eegprep_lean`, with an empty dependency list. NEMAR keeps importing `eegprep_lean`, including `set_default_transport`. Neither distribution depends on the other. This repository does not add `eegprep-lean` to its dependencies, and the lean wheel does not depend on `eegprep`.

The lean wheel becomes a thin alias of `eegprep` only after the base wheel is actually small. Publishing that alias earlier would replace the vendored 28,537-byte install with the scientific install. OSA vendors `eegprep_lean-0.1.0.dev3-py3-none-any.whl` and checks its sha256. A rebuilt wheel is a pin someone bumps on purpose.

## What this proposal does not do yet

Moving oct2py, psutil, and pyedflib to extras named `eeglab`, `sys`, and `edf` can land before the major version. Until the major version, `eegprep[full]` restores those three packages, and numpy, scipy, mne, matplotlib, pybids, python-picard, and h5py stay required. At the major version, pyedflib joins `io` and `full` grows to the union in the table. This document does not make that split the current install.

## Why the base has to shrink in a major version

An extra is added on top of the base requirements. `pip install eegprep[full]` on today's base still downloads today's base. ADR 0069 recorded that limit and kept the default install whole so existing users keep mne, matplotlib, pybids, picard, h5py, and oct2py. Shrinking that list changes what `pip install eegprep` means. That is a major-version break, with `eegprep[full]` as the compatible install.

## Measurements the decision uses

On current `develop`, the base lock closure is 83 packages and 136.4 MB. Dropping oct2py, psutil, and pyedflib leaves 55 packages and 118.5 MB. numpy, scipy, pyyaml, packaging, colorama, and threadpoolctl together are 6 packages and 54.8 MB. Those four pure-Python packages without numpy and scipy are 4 packages and 1.0 MB. The base plus the gui extra is about 397 MB. The base plus the torch extra is about 2.1 GB.

The largest wheels in the base closure are scipy 37.3 MB, numpy 16.5 MB, pandas 12.0 MB, scikit-learn 9.5 MB, matplotlib 8.8 MB, pillow 8.1 MB, mne 7.5 MB, debugpy 5.3 MB, fonttools 5.0 MB, and h5py 4.9 MB. Several of those arrive through another package:

- oct2py depends on octave-kernel, which depends on ipykernel and metakernel, which bring ipython, jedi, debugpy, and pyzmq.
- python-picard depends on scikit-learn.
- pybids depends on pandas, nibabel, sqlalchemy, and formulaic.
- mne depends on matplotlib. matplotlib is also a direct base requirement today, so taking mne out of the base leaves matplotlib in place until `plot` is its own extra.

ADR 0069 measured a different target in September 2026: under Pyodide 0.29.5 the eegprep base was 61 packages and 62.6 MB, against 12 packages and 4.2 MB to read a window. Both figures are real. Pyodide already carries numpy, scipy, matplotlib, and h5py, so its incremental download is smaller than a desktop pip install of the same names.

`import eegprep` is already lazy. In an empty environment its median import time was 26 ms, and it loaded none of numpy, scipy, mne, matplotlib, h5py, oct2py, torch, or PySide6. `import eegprep_lean` had a median of 84 ms. `dir(eegprep)` listed 19 names while `__all__` had 539, which is the discoverability gap `help()` and `__dir__` close. Issue #428 and pull request #440 added that behavior to `eegprep-lean` only.

The published `eegprep` 0.4.0 wheel is 16,343,718 bytes. Inside it, `netICL.mat` is 10.82 MB, `standard_mri_mni.npz` is 3.84 MB, `mheadnew.mat` is 0.37 MB, all `.py` files together are 4.00 MB, and all `.mat` files together are 11.21 MB. A base install whose dependency list is numpy and scipy still ships a 16 MB wheel while those data files stay in the package. The `eegprep-lean` wheel OSA vendors is 28,537 bytes, version `0.1.0.dev3`, and it is not on PyPI. `depends.toml` in that runtime expects matplotlib, numpy, and zarr beside it. The prelude is `eegprep_lean.set_default_transport(eegprep_lean.FetchTransport(osa.fetch))`.

Issue #395 is the check to repeat for every extra: eeglabio had no import site in this repository and was still required, because mne imports it when writing an EEGLAB file. A missing extra has to be caught by importing the feature with the library absent, not by grepping for the import name.

Pyodide floors in `tests/test_browser_dependency_floors.py` stay where they are: numpy 2.2.5, scipy 1.14.1, matplotlib 3.8.4, h5py 3.13.0. `requires-python >= 3.12` still admits Pyodide 3.13.2.

## Open questions

Arno:

- Does file I/O (`h5py`, `eeglabio`, `pyedflib`) stay in the required install, or move to the `io` extra? Loading a `.set` file is the first thing most scripts do. The table above puts it in `io`. If the required install should open EEGLAB files, `io` joins numpy and scipy and `full` still covers the rest.
- Do `netICL.mat` (10.82 MB) and the dipfit MRI (`standard_mri_mni.npz`, 3.84 MB) leave the base wheel with the `iclabel` extra? Together with `mheadnew.mat` they are most of the 16.3 MB wheel. Leaving them in the base wheel keeps the package file large after the dependency list shrinks.

Seyed:

- Confirm the OSA pin before any lean wheel is rebuilt. The vendored file is `eegprep_lean-0.1.0.dev3-py3-none-any.whl`, 28,537 bytes, and `depends.toml` expects matplotlib, numpy, and zarr. The contract text in nemar-cli still describes `0.1.0.dev2`.
- Is `eegprep-lean` ever published to PyPI? OSA installs from its own lock and does not need PyPI. The lean wheel workflow leaves that choice to the project owner.
- When this proposal is accepted, update the nemar-cli contract so it matches this document. The alias stays unpublished until the base wheel is small enough that depending on `eegprep` preserves the browser download. With numpy and scipy required, that dependency closure is about 55 MB of wheels, and the wheel file stays about 16 MB until the data files move.
