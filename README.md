# EEGPrep

EEGPrep is a Python package that reproduces the EEGLAB default preprocessing pipeline with numerical accuracy down to 10⁻⁷, including clean_rawdata and ICLabel, enabling MATLAB-to-Python equivalence for EEG analysis. The package is fully documented for conversion, packaging, and testing workflows, with installation available via PyPI.

## Install

```
pip install eegprep
```

# Current code coverage

<img width="489" height="1119" alt="Screenshot 2025-08-21 at 09 15 37" src="https://github.com/user-attachments/assets/cb958237-16bb-4f57-867b-d2cd393a42a2" />

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
