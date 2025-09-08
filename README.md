# EEGPrep

EEGPrep is a Python package that reproduces the EEGLAB default preprocessing pipeline with numerical accuracy down to 10⁻⁷, including clean_rawdata and ICLabel, enabling MATLAB-to-Python equivalence for EEG analysis. It takes BIDS data as input and produces BIDS derivative dataset as output, which can then be reimported into other packages as needed (EEGLAB, Fieldtrip, Brainstorm, MNE). It does produce plots. The package will be fully documented for conversion, packaging, and testing workflows, with installation available via PyPI.

## Pre-release

EEGPrep is currently in a pre-release phase. It functions end-to-end (bids branch) but has not yet been tested with multiple BIDS datasets. The documentation is incomplete, and use is at your own risk. The planned release is scheduled for the end of 2025.

## Install

```
pip install eegprep
```

# Current code coverage

This is the current coverage of the test cases. The goal is to achieve 90% coverage.

<img width="451" height="1353" alt="Screenshot 2025-08-24 at 09 17 53" src="https://github.com/user-attachments/assets/99ac7fa6-c467-4523-94a0-a368af5b0de6" />

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

Use tests under Cursos or Visual Studio Code.

## Core maintainers

- Arnaud Delorme, UCSD, CA, USA
- Christian Kothe, Intheon, CA, USA
- Bruno Aristimunha Pinto, Inria, France
