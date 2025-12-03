<img src="https://github.com/user-attachments/assets/d7230b12-7fb8-4abb-94a0-33c47286b019" width="300">

# What is EEGPrep?

[![Documentation Status](https://github.com/sccn/eegprep/actions/workflows/docs.yml/badge.svg)](https://github.com/sccn/eegprep/actions/workflows/docs.yml)
[![GitHub Pages](https://github.com/sccn/eegprep/actions/workflows/pages.yml/badge.svg)](https://sccn.github.io/eegprep/)

EEGPrep is a Python package that reproduces the EEGLAB default preprocessing pipeline with numerical accuracy down to 1e-5 uV, including clean_rawdata and ICLabel, enabling MATLAB-to-Python equivalence for EEG analysis. It takes BIDS data as input and produces BIDS derivative dataset as output, which can then be reimported into other packages as needed (EEGLAB, Fieldtrip, Brainstorm, MNE). It does produce plots. The package will be fully documented for conversion, packaging, and testing workflows, with installation available via PyPI.

**📚 [View Full Documentation](https://sccn.github.io/eegprep/)** | **🔧 [GitHub Pages Setup Guide](docs/GITHUB_PAGES_SETUP.md)**

## Pre-Release

EEGPrep is currently in a pre-release phase. It functions end-to-end (bids branch) but has not yet been tested with multiple BIDS datasets. The documentation is incomplete, and use is at your own risk. The planned release is scheduled for the end of 2025.


## Install

To install the complete EEGPrep including the ICLabel classifier (which can pull in ~7GB of binaries on Linux), use the following line:
```
pip install eegprep[all]
```

To install the lean version:
```
pip install eegprep
```

You can then manually install a lightweight CPU-only version of PyTorch if desired by 
your operating system.


# Comparing MATLAB and Python implementations

The MATLAB and Python implementations were compared using the first two subjects from the BIDS datasets [ds003061](https://nemar.org/dataexplorer/detail?dataset_id=ds003061) and [ds002680](https://nemar.org/dataexplorer/detail?dataset_id=ds002680) available on NEMAR. The observed differences were extremely small, with the largest (during HighpassFilter) below 0.002, indicating excellent numerical consistency between the two implementations.

<img width="1744" height="1049" alt="Screenshot 2025-10-02 at 11 43 03" src="https://github.com/user-attachments/assets/79c17151-e2e3-4acc-b144-accdf34ae4c5" />

# versioning
- Change version inside the file pyproject.toml
- Change version inside the file main (for docker)
- Run make_release in the script folder and tag with the version
- Use the correct docker version when building (see below)

# Docker (SCCN Power Users)

```
docker build -t eegprep:0.2.9 -f DOCKERFILE .
docker tag eegprep:0.2.9 arnodelorme/eegprep:0.2.9
docker push arnodelorme/eegprep:0.2.9
```

Check the project on https://hub.docker.com/

Mounted folder in /usr/src/project

# PYPI Release Process (Maintainers Only)

## Quick Release Workflow

Use the release script for streamlined releases:

```bash
python scripts/make_release.py
```

The script will:
1. Check prerequisites (build, twine, git status)
2. Confirm the version from `pyproject.toml`
3. Let you choose: test release, production release, or both
4. Build and upload the package (automatically uses `eegprep_test` name for TestPyPI)
5. Create and push git tags for production releases

> **Note:** The script automatically handles a TestPyPI naming conflict by building a package
> with the name `eegprep_test` for test releases.

## Prerequisites

Install build tools:
```bash
pip install build twine
```

## API Tokens
- Get API token for PyPI and TestPyPI (both maintainers should have these)
- Twine will prompt for them during upload
- Store them in `~/.pypirc` for convenience

## Manual Release Process

**Recommended:** Use `scripts/make_release.py` instead to avoid manual errors with package naming.

If you need to release manually:

**1. Update version in `pyproject.toml`**

**2. Test release (staging):**

> **Note:** A former maintainer owns the `eegprep` package name on TestPyPI, so you will not be able to post a 
> package named `eegprep` there at this time. 
> To work around this when performing the build manually (note the `make_release.py` script handles this for you), temporarily change the package name to `eegprep_test` in `pyproject.toml` before building.
> Remember to change it back to `eegprep` after uploading!

```bash
# In pyproject.toml, temporarily change: name = "eegprep" to name = "eegprep_test"
python -m build
python -m twine upload --repository testpypi dist/*
# Change name back to "eegprep" in pyproject.toml

# Test the installation:
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ eegprep_test==X.Y.Z
# (imports still work as 'import eegprep')
```

**3. Production release:**
```bash
python -m twine upload dist/*
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
pip install eegprep==X.Y.Z
```

## Documentation
https://packaging.python.org/en/latest/tutorials/packaging-projects/

## Install Package
Packaging was done following the tutorial: https://packaging.python.org/en/latest/tutorials/packaging-projects/ with setuptools

To install the package with all optional dependencies, run:
```
pip install eegprep[all]
```

## Running Tests

Install MATLAB interface `pip install /your/path/to/matlab/extern/engines/python` (for example on OSx `pip install /Applications/MATLAB_R2025a.app/extern/engines/python 
Processing /Applications/MATLAB_R2025a.app/extern/engines/python`)

Check installation

```python
import matlab.engine
engine = matlab.engine.start_matlab()
engine.eval("disp('hello world');", nargout=0)
```

Use tests/main_compare.m

This project uses `unittest`. You can run tests from the project root via the command:
```
python -m unittest discover -s tests
```
...or use the unittest integration in your IDE (e.g., PyCharm, VS Code, or Cursor).

## Core maintainers

- Arnaud Delorme, UCSD, CA, USA
- Christian Kothe, Intheon, CA, USA
- Bruno Aristimunha Pinto, Inria, France
