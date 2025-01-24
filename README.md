## Branlife execution

Simply press the Execute button. The documentation below is related to the EEGprep GitHub repository.

## Build docker

docker run --rm -it -v $(pwd):/usr/src/project dtyoung/eegprep /bin/bash
docker run -u root --rm -it -v $(pwd):/usr/src/project dtyoung/eegprep /bin/bash

# remove

docker rmi dtyoung/eegprep

Mounted folder in /usr/src/project

## How to convert a function from MATLAB to Python

1. Get a MATLAB file to load and process an EEG file

2. Convert the code using GPT4 or when short Copilot and test in a Notebook. Once the code runs without erroring, move to 3.

3. Use the Jupyter code to create a Python file (not notebook) to load the same file as MATLAB and process it as well (in plain Python, not in a subfunction)

4. Start the debugger in both and compare. Note that it is better to use the debugger on Python file than Jupyter Notebook (could not get it to stop)

5. Once the result is the same, package the Python code in a function with the same name as MATLAB

6. Write the function to compare (see example) and the helper Python function to load the file (note that there could be a general Python helper function)

## Create package

# Documentation
https://packaging.python.org/en/latest/tutorials/packaging-projects/

# API tokens
- Get API token, one for official and one for test(Dung has it)
- Twine will ask them from you

# Update version

Change version in pyproject.toml

# Staging release
```
python -m build
python -m twine upload --repository testpypi dist/*
```

to test
```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ eegprep==0.0.x
```

# Final release
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
pip install eegprep==0.0.2
```

## Test

use tests/main_compare.m

## Versions

0.1 - Initial BrainLife
