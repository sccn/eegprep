# EEGPrep

EEGPrep is a Python package that reproduces the EEGLAB default preprocessing pipeline with numerical accuracy down to 10⁻⁷, including clean_rawdata and ICLabel, enabling MATLAB-to-Python equivalence for EEG analysis. It takes BIDS data as input and produces BIDS derivative dataset as output, which can then be reimported into other packages as needed (EEGLAB, Fieldtrip, Brainstorm, MNE). It does produce plots. The package will be fully documented for conversion, packaging, and testing workflows, with installation available via PyPI.

## Pre-release

EEGPrep is currently in a pre-release phase. It functions end-to-end (bids branch) but has not yet been tested with multiple BIDS datasets. The documentation is incomplete, and use is at your own risk. The planned release is scheduled for the end of 2025.

## Install

```
pip install eegprep
```

# Comparing MATLAB and Python implementations

The MATLAB and Python implementations show extremely small differences across all stages, with the largest discrepancy (HighpassFilter) below 0.002. Overall, the results are nearly identical, indicating excellent numerical consistency between the two implementations.

<img width="1744" height="1049" alt="Screenshot 2025-10-02 at 11 43 03" src="https://github.com/user-attachments/assets/79c17151-e2e3-4acc-b144-accdf34ae4c5" />

# Docker

## Build Docker

```
docker run --rm -it -v $(pwd):/usr/src/project dtyoung/eegprep /bin/bash
docker run -u root --rm -it -v $(pwd):/usr/src/project dtyoung/eegprep /bin/bash
```

## Remove Docker

docker rmi dtyoung/eegprep

Mounted folder in /usr/src/project

# Pypi release notes

## Documentation
https://packaging.python.org/en/latest/tutorials/packaging-projects/

## API tokens
- Get API token, one for official and one for test(Dung has it)
- Twine will ask them from you

## Update version

Change version in pyproject.toml

## Staging release
```
python -m build
python -m twine upload --repository testpypi dist/*
```

to test
```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ eegprep==0.0.x
```

## Final release
```
twine upload dist/*
```

to test

```
pip install eegprep
```

## Install package
Packaging was done following the tutorial: https://packaging.python.org/en/latest/tutorials/packaging-projects/ with setuptools

To install the package, run:
```
pip install eegprep
```

## Test

Install MATLAB insterface "pip install /your/path/to/matlab/extern/engines/python"
Use tests/main_compare.m

Use tests under Cursos or Visual Studio Code.

## Core maintainers

- Arnaud Delorme, UCSD, CA, USA
- Christian Kothe, Intheon, CA, USA
- Bruno Aristimunha Pinto, Inria, France
