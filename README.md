<p align="center">
  <img src="https://github.com/user-attachments/assets/d7230b12-7fb8-4abb-94a0-33c47286b019" width="300" alt="EEGPrep logo">
</p>

# EEGPrep

[![Tests](https://github.com/sccn/eegprep/actions/workflows/test.yml/badge.svg)](https://github.com/sccn/eegprep/actions/workflows/test.yml)
[![Documentation Status](https://github.com/sccn/eegprep/actions/workflows/docs.yml/badge.svg)](https://github.com/sccn/eegprep/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

EEGPrep is an EEGLAB-compatible, Python-native toolkit for loading, cleaning,
visualizing, scripting, and validating EEG preprocessing workflows. It keeps the
EEGLAB concepts researchers already know, including `EEG`, `ALLEEG`,
`CURRENTSET`, `pop_*` functions, command history, ICA fields, channel locations,
events, epochs, STUDY workflows, and EEGBrowser-style review, while providing a
standalone Python package with GUI, interactive console, CLI, tests, and Sphinx
documentation.

EEGPrep is built for EEG researchers who want EEGLAB-style workflows, Python
reproducibility, and agent-friendly automation without requiring MATLAB or an
EEGLAB checkout at runtime.

**Documentation:** [sccn.github.io/eegprep](https://sccn.github.io/eegprep/) |
**Issue tracker:** [github.com/sccn/eegprep/issues](https://github.com/sccn/eegprep/issues)

## Why EEGPrep?

- **Familiar to EEGLAB users.** Function names, data structures, menus, history
  commands, and GUI workflows follow EEGLAB where that helps users move between
  tools.
- **Python-native and standalone.** Runtime code lives in the `eegprep` package
  and works without calling MATLAB or reading the vendored EEGLAB reference tree.
- **GUI and console share one workspace.** Launch `eegprep-console` and switch
  between the Qt GUI and Python commands while `EEG`, `ALLEEG`, `CURRENTSET`,
  `LASTCOM`, `ALLCOM`, `STUDY`, and `CURRENTSTUDY` stay synchronized.
- **Scriptable from day one.** The same `pop_*` workflows can be used from the
  GUI, Python scripts, and the command-line interface.
- **Designed for reproducible research.** CLI commands support structured JSON,
  manifests, pipeline validation, stable error codes, and bundled agent guidance.
- **Validated against EEGLAB.** Numerical parity tests and visual parity checks
  compare EEGPrep behavior with EEGLAB for deterministic workflows.

## Install

EEGPrep uses `uv` for development and CI. For a published release, install the
lean package with:

```bash
uv add eegprep
```

To install all optional extras, including GUI, console, docs, and classifier
dependencies, use:

```bash
uv add "eegprep[all]"
```

If you are installing into a non-`uv` environment, `pip install eegprep` remains
supported for published releases:

```bash
pip install eegprep
```

For a source checkout:

```bash
git clone https://github.com/sccn/eegprep.git
cd eegprep
uv sync --group dev
```

The complete install can pull in large optional binaries on some platforms,
especially for ICLabel/PyTorch support. You can install a lightweight CPU-only
PyTorch build manually if that better matches your system.

## Quick Start

The repository includes tutorial data in `sample_data/`, named after EEGLAB's
sample-data convention.

### Shared GUI and Python Console

Launch EEGPrep with the main GUI and a synchronized IPython workspace:

```bash
uv run eegprep-console --full
```

Then use the GUI to load `sample_data/eeglab_data.set`, run preprocessing
actions from the menus, and inspect the same state from the console:

```python
EEG["setname"], EEG["srate"], EEG["nbchan"], EEG["pnts"]
CURRENTSET
LASTCOM
eegh()
```

You can also run `pop_*` commands directly:

```python
pop_resample(EEG, 64)
pop_reref(EEG, [])
```

GUI actions and console commands append to the same history, so workflows can be
replayed or moved into scripts.

### GUI Only

```bash
uv run eegprep-gui --full
```

Use this when you want the EEGLAB-style menu workflow without an attached
console.

### Agent-Friendly CLI

The `eegprep` command is intended for headless pipelines, batch processing, and
AI agents that need machine-readable output.

```bash
uv run eegprep inspect sample_data/eeglab_data.set --json
uv run eegprep validate sample_data/eeglab_data.set --json
uv run eegprep capabilities --json
uv run eegprep skills get eegprep-cli
```

Transform commands use the same EEGPrep processing functions as the Python API
and write manifests for reproducibility:

```bash
uv run eegprep resample sample_data/eeglab_data.set \
  --freq 64 \
  --output sample_data/eeglab_data_64hz.set \
  --manifest sample_data/eeglab_data_64hz_manifest.json \
  --json
```

### Python API

```python
from pathlib import Path

from eegprep import pop_eegfiltnew, pop_loadset, pop_resample, pop_saveset

input_file = Path("sample_data") / "eeglab_data.set"
output_file = Path("sample_data") / "eeglab_data_quickstart.set"

EEG = pop_loadset(input_file)
EEG, filter_com = pop_eegfiltnew(
    EEG,
    locutoff=1.0,
    hicutoff=40.0,
    plotfreqz=False,
    return_com=True,
)
EEG, resample_com = pop_resample(EEG, 64, return_com=True)
pop_saveset(EEG, output_file)

print(filter_com)
print(resample_com)
```

`return_com=True` returns the updated dataset and the replayable command string
that EEGPrep records in GUI and console history.

## For EEGLAB Users

| EEGLAB concept | EEGPrep equivalent |
| --- | --- |
| `eeglab` GUI | `uv run eegprep-gui --full` or `uv run eegprep-console --full` |
| `EEG`, `ALLEEG`, `CURRENTSET` | Shared `EEGPrepSession` state in the GUI and console |
| MATLAB command history | `LASTCOM`, `ALLCOM`, and `eegh()` in `eegprep-console` |
| `pop_*` wrappers | Python `eegprep.pop_*` functions with `return_com=True` |
| EEGBrowser / scrolling review | EEGPrep EEGBrowser and `pop_eegplot` workflows |
| clean_rawdata, ICLabel, FIRFilt, DIPFIT, EEG-BIDS | Bundled EEGPrep plugin ports and menu integrations |
| STUDY workflows | Python STUDY structures, GUI paths, and `std_*` helpers |
| MATLAB scripts | Python scripts, CLI pipelines, and history-derived commands |

EEGPrep follows EEGLAB's one-based user-facing indices where researchers expect
them, while using zero-based Python indices internally. Continuous data is
channel-major, usually `(nbchan, pnts)`, and epoched data is usually
`(nbchan, pnts, trials)`.

## What Is Included

- EEGLAB `.set` loading/saving, BIDS import/export, and common EEG file I/O.
- BIDS derivative workflows intended to interoperate with EEGPrep, EEGLAB,
  FieldTrip, Brainstorm, MNE, and other EEG analysis tools.
- Dataset, event, channel-location, epoch, and history workflows.
- Filtering, resampling, rereferencing, cleaning, rejection, interpolation, ICA,
  component review, ICLabel, DIPFIT, topographies, spectra, time-frequency, and
  statistics workflows.
- EEGBrowser-style scrolling inspection, marking, and rejection.
- STUDY and group-level workflow support.
- Extension SDK, plugin discovery, extension validation, and agent-facing
  extension authoring guidance.
- Sphinx documentation, API reference, examples, and a structured CLI.

## Documentation

Start with the [online documentation](https://sccn.github.io/eegprep/):

- [Quick Start](https://sccn.github.io/eegprep/user_guide/quickstart.html)
- [GUI and Console Session](https://sccn.github.io/eegprep/user_guide/gui_console_session.html)
- [Scripting Workflows](https://sccn.github.io/eegprep/user_guide/scripting_workflows.html)
- [EEGBrowser](https://sccn.github.io/eegprep/user_guide/eegbrowser.html)
- [Extensions](https://sccn.github.io/eegprep/user_guide/extensions.html)
- [API Reference](https://sccn.github.io/eegprep/api/index.html)

Repository documentation helpers are kept in [`docs/`](docs/). The docs
workflow publishes the Sphinx site to GitHub Pages.

## Project Status

EEGPrep is in active pre-release development. It is intended to become a
standalone Python counterpart for core EEGLAB preprocessing workflows. Current
development emphasizes EEGLAB parity, GUI and console usability, BIDS support,
extension support, and reproducible headless workflows.

Use EEGPrep with the same care you would apply to any research preprocessing
software: validate pipelines on representative data, inspect intermediate
outputs, and record the exact version and command history used for analysis.

## Numerical Parity

EEGPrep is developed against EEGLAB as a parity oracle. The MATLAB and Python
implementations are tested for close numerical agreement, including default
preprocessing pipeline comparisons that target accuracy down to `1e-5` uV where
the algorithms are deterministic. They have also been compared using the first
two subjects from the BIDS datasets
[ds003061](https://nemar.org/dataexplorer/detail?dataset_id=ds003061) and
[ds002680](https://nemar.org/dataexplorer/detail?dataset_id=ds002680) on NEMAR.
Observed differences were very small, with the largest reported HighpassFilter
difference below `0.002`, indicating strong numerical consistency for the tested
workflows.

<img width="1744" height="1049" alt="MATLAB and Python implementation comparison" src="https://github.com/user-attachments/assets/79c17151-e2e3-4acc-b144-accdf34ae4c5" />

## Development

Run tests from the project root with:

```bash
uv run pytest tests
```

For quick local iteration:

```bash
uv run pytest -m "not slow"
```

The repo uses `ruff`, `ty`, and `pre-commit.py` for linting, formatting, and
type-checking:

```bash
./pre-commit.py --changed-from origin/develop
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
```

MATLAB parity tests require MATLAB Engine for Python. Install the engine from
your MATLAB installation, for example on macOS:

```bash
uv pip install /Applications/MATLAB_R2025a.app/extern/engines/python
```

Check the installation:

```python
import matlab.engine

engine = matlab.engine.start_matlab()
engine.eval("disp('hello world');", nargout=0)
```

The MATLAB comparison entry point is `tests/matlab/main_compare.m`.

## Cite

If EEGPrep contributes to your research, please cite EEGPrep and the EEGLAB
methods it ports. For EEGLAB, cite:

Delorme, A., & Makeig, S. (2004). EEGLAB: an open source toolbox for analysis
of single-trial EEG dynamics including independent component analysis. *Journal
of Neuroscience Methods*, 134(1), 9-21.

## Core Maintainers

- Arnaud Delorme, UCSD, CA, USA
- Suraj Ranganath, UCSD, CA, USA
- Christian Kothe, Intheon, CA, USA
- Bruno Aristimunha Pinto, Inria, France

<details>
<summary>Maintainer release notes</summary>

### Release Process

Releases are published by `.github/workflows/release.yml` when a `v*` tag is
pushed. That is the only path to PyPI. Full instructions, including the dry run,
are in [docs/source/releasing.rst](docs/source/releasing.rst).

The version lives in `src/eegprep/__init__.py` (`__version__`); `pyproject.toml`
reads it via `dynamic = ["version"]`, so there is nothing to edit there.

```bash
# dry run first: everything except publishing
gh workflow run release.yml --ref master -f dry_run=true

# bump __version__, land it on master, then tag to release
git commit -am "release: 0.3.0"
git push origin develop:master
git tag -a v0.3.0 -m "Release version 0.3.0" && git push origin v0.3.0
```

The workflow lints, type-checks, runs the test suite, builds, verifies the
artifacts, publishes to PyPI with Trusted Publishing (no stored token), and
creates the GitHub release.

CI does not build Docker images. After the tag is published:

```bash
docker login
uv run python scripts/build_docker.py
```

That builds and pushes `arnodelorme/eegprep:<version>` and updates the image pin
in `tools/hpc/main.pbs`; commit the pin.

Verify the published release:

```bash
uv pip install eegprep==X.Y.Z
```

Packaging follows the Python Packaging User Guide:
<https://packaging.python.org/en/latest/tutorials/packaging-projects/>.

</details>

<details>
<summary>Docker notes for SCCN power users</summary>

```bash
docker build -t eegprep:0.2.9 -f DOCKERFILE .
docker tag eegprep:0.2.9 arnodelorme/eegprep:0.2.9
docker push arnodelorme/eegprep:0.2.9
```

Check the project on Docker Hub: <https://hub.docker.com/>.

Mounted folders are available in `/usr/src/project`.

</details>
