## How to convert a function from MATLAB to Python

1. Get a MATLAB file to load and process an EEG file

2. Convert the code using GPT4 or when short Copilot and test in a Notebook. Once the code runs without erroring, move to 3.

3. Use the Jupyter code to create a Python file (not notebook) to load the same file as MATLAB and process it as well (in plain Python, not in a subfunction)

4. Start the debugger in both and compare. Note that it is better to use the debugger on Python file than Jupyter Notebook (could not get it to stop)

5. Once the result is the same, package the Python code in a function with the same name as MATLAB

6. Write the function to compare (see example) and the helper Python function to load the file (note that there could be a general Python helper function)

## Install package
Packaging was done following the tutorial: https://packaging.python.org/en/latest/tutorials/packaging-projects/ with setuptools

To install the package, run:
```
pip install eegprep==0.0.2
```
