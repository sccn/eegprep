# 0071. The next major version installs a zero-dependency base

- Status: Proposed
- Date: 2026-11-03
- Supersedes: the "default install stays whole" clause of [ADR 0069](https://github.com/nemarOrg/nemar-cli/blob/main/.context/decisions/0069-the-browser-runtime-is-its-own-package-and-eegprep-stays-whole.md) (accepted 2026-09-20, owner Seyed Yahya Shirazi, in nemar-cli). That clause keeps every current dependency on `pip install eegprep`.
- Leaves in force until the alias below: the rest of ADR 0069. `eegprep-lean` remains its own distribution while NEMAR pins it. An extra can only add packages, so it cannot shrink the base install. ADR 0070, the browser recipe, is unchanged.
- Number: 0071, so this file does not collide with ADR 0070. It lives in this repository because `.context/` is gitignored here.

Suraj chose this shape. A plain `pip install eegprep` should be as small as `eegprep-lean`: no required dependencies, and a wheel on the order of the vendored lean wheel (28,537 bytes). Preprocessing and the rest of today's stack are named extras. `eegprep[full]` restores today's install. A missing extra raises `ImportError` with the command `pip install "eegprep[<extra>]"`.

This is a major version bump. Arno and Seyed still have the questions at the end, so this document is a proposal and the current install does not change.

Measurements below are from the feasibility report on `develop` at `f22111a5` (pull request #517), walked from `uv.lock` for linux x86_64, Python 3.12. Sizes are compressed wheel bytes, which is what pip downloads.

## Decision

In the next major version:

- `pip install eegprep` has an empty dependency list. The install is the package wheel alone, the same idea as `eegprep-lean` (28 KB, no dependencies). It includes the lazy public names, `eegprep.help()`, and the NEMAR reader once that source lives in this tree.
- `eegprep[core]` adds numpy and scipy, plus the small pure-Python helpers preprocessing still imports (`pyyaml`, `packaging`, `threadpoolctl`, and `colorama` if logging still needs it).
- The other features stay named extras. `eegprep[full]` is their union with `core`, which is today's install.

| Extra | Installs | What it is for |
| --- | --- | --- |
| `core` | numpy, scipy, and the small helpers above | Preprocessing |
| `io` | h5py, eeglabio | Reading and writing EEG files other than EDF/BDF |
| `edf` | pyedflib | EDF and BDF output |
| `plot` | matplotlib | Topography and other figures |
| `bids` | pybids | BIDS import and export |
| `ica` | python-picard | Picard ICA |
| `mne` | mne | Conversion to and from MNE |
| `eeglab` | oct2py | The Octave engine |
| `sys` | psutil | RAM reservations for ASR and job planning |
| `iclabel` | ONNX runtime and the model file | ICLabel. The network weights and the dipfit MRI leave the base wheel with this extra |
| `gui` | unchanged | Qt dialogs |
| `console` | unchanged | `eegprep-console` |
| `torch` | unchanged | Torch models |
| `docs` | unchanged | The documentation build |
| `full` | the union of the extras above | Today's install |

A script that today runs after `pip install eegprep` runs after `pip install eegprep[full]`.

Calling a function whose library is missing raises `ImportError`. The message names the library, the function or feature, the command `pip install "eegprep[<extra>]"`, and `pip install "eegprep[full]"`. `help()` prints the same extra next to each public name, read from source, without importing the module.

Empty dependency metadata is not, by itself, a 28 KB wheel. The published `eegprep` 0.4.0 wheel is 16,343,718 bytes. `netICL.mat` is 10.82 MB, `standard_mri_mni.npz` is 3.84 MB, and the Python sources together are 4.00 MB. The network and the MRI have to leave the base wheel, and the built base wheel has to come down to the lean wheel's size, before the two installs match. The alias in the next section waits on that built wheel.

## eegprep-lean

NEMAR and OSA pin `eegprep_lean-0.1.0.dev3-py3-none-any.whl` (28,537 bytes) by sha256. That pin keeps working. Until the eegprep base actually matches lean, this repository keeps building a separate zero-dependency wheel from one shared copy of the reader source, under the import name `eegprep_lean`, including `set_default_transport`. Neither distribution depends on the other during that time.

`eegprep-lean` becomes an alias that installs `eegprep` only after that base matches: no required dependencies, and a wheel as small as the one OSA vendors. Publishing the alias earlier would replace the 28 KB pin with the scientific install.

## What can land before the major version

oct2py, psutil, and pyedflib can move to extras named `eeglab`, `sys`, and `edf` without a major bump. Until the major version, `eegprep[full]` in that change restores those three packages, and numpy, scipy, mne, matplotlib, pybids, python-picard, and h5py stay required. The major version is the point at which the base dependency list becomes empty and `full` grows to the union in the table.

## Why this is a major version

An extra is added on top of the base requirements. `pip install eegprep[full]` on today's base still downloads today's base. ADR 0069 recorded that limit and kept the default install whole so existing users keep mne, matplotlib, pybids, picard, h5py, and oct2py. An empty base list changes what `pip install eegprep` means. That break belongs in a major version, with `eegprep[full]` as the compatible install.

## Considered alternative

The alternative is a required base of numpy and scipy (plus the small helpers), with everything else optional. That install is still much smaller than today: the lock walk puts those packages at 6 wheels and 54.8 MB, against 83 packages and 136.4 MB for the current base.

It is not the recommendation. scipy's own wheel is 37.3 MB, about 35–40 MB, and numpy adds 16.5 MB. A plain install of that size cannot replace the lean wheel. The alias would hand NEMAR a ~55 MB download in place of 28 KB, so `eegprep-lean` would have to stay a separate distribution permanently. The zero-dependency base is what makes retiring that wheel possible.

## Measurements the decision uses

On current `develop`, the base lock closure is 83 packages and 136.4 MB. Dropping oct2py, psutil, and pyedflib leaves 55 packages and 118.5 MB. numpy, scipy, pyyaml, packaging, colorama, and threadpoolctl together are 6 packages and 54.8 MB. Those four pure-Python packages without numpy and scipy are 4 packages and 1.0 MB. The base plus the gui extra is about 397 MB. The base plus the torch extra is about 2.1 GB.

The largest wheels in the base closure are scipy 37.3 MB, numpy 16.5 MB, pandas 12.0 MB, scikit-learn 9.5 MB, matplotlib 8.8 MB, pillow 8.1 MB, mne 7.5 MB, debugpy 5.3 MB, fonttools 5.0 MB, and h5py 4.9 MB. Several of those arrive through another package:

- oct2py depends on octave-kernel, which depends on ipykernel and metakernel, which bring ipython, jedi, debugpy, and pyzmq.
- python-picard depends on scikit-learn.
- pybids depends on pandas, nibabel, sqlalchemy, and formulaic.
- mne depends on matplotlib. matplotlib is also a direct base requirement today, so taking mne out of the base leaves matplotlib in place until `plot` is its own extra.

ADR 0069 measured a different target in September 2026: under Pyodide 0.29.5 the eegprep base was 61 packages and 62.6 MB, against 12 packages and 4.2 MB to read a window. Both figures are real. Pyodide already carries numpy, scipy, matplotlib, and h5py, so its incremental download is smaller than a desktop pip install of the same names.

`import eegprep` is already lazy. In an empty environment its median import time was 26 ms, and it loaded none of numpy, scipy, mne, matplotlib, h5py, oct2py, torch, or PySide6. `import eegprep_lean` had a median of 84 ms. `dir(eegprep)` listed 19 names while `__all__` had 539. Issue #428 and pull request #440 added `help()` and `__dir__` to `eegprep-lean` only.

Inside the published wheel, `mheadnew.mat` is 0.37 MB, all `.py` files together are 4.00 MB, and all `.mat` files together are 11.21 MB. The `eegprep-lean` wheel is not on PyPI. `depends.toml` in the OSA runtime expects matplotlib, numpy, and zarr beside it. The prelude is `eegprep_lean.set_default_transport(eegprep_lean.FetchTransport(osa.fetch))`.

Issue #395 is the check to repeat for every extra: eeglabio had no import site in this repository and was still required, because mne imports it when writing an EEGLAB file. A missing extra has to be caught by importing the feature with the library absent.

Pyodide floors in `tests/test_browser_dependency_floors.py` stay where they are: numpy 2.2.5, scipy 1.14.1, matplotlib 3.8.4, h5py 3.13.0. `requires-python >= 3.12` still admits Pyodide 3.13.2.

## Open questions

Suraj has asked Arno:

1. Is a minimal plain install acceptable? The proposal is `pip install eegprep` with no required dependencies, at the size of `eegprep-lean`.
2. Should the ICLabel network (about 11 MB, measured 10.82 MB in `netICL.mat`) and the dipfit MRI (about 4 MB, measured 3.84 MB in `standard_mri_mni.npz`) leave the base wheel?
3. Is the major version bump acceptable? `eegprep[full]` would be the install that matches today's `pip install eegprep`.

Seyed:

- The OSA pin stays on `eegprep_lean-0.1.0.dev3-py3-none-any.whl`, 28,537 bytes, checked by sha256, until the eegprep base matches that wheel and the alias is agreed. `depends.toml` expects matplotlib, numpy, and zarr beside it. Any rebuilt lean wheel is a pin he bumps on purpose.
- When this proposal is accepted, the nemar-cli contract should be updated to match it. The contract text still describes `0.1.0.dev2`. The alias stays unpublished until the built base wheel matches the lean wheel those documents pin.
