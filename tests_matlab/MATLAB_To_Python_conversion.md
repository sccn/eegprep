# Developer's notes

## How to convert a function from MATLAB to Python

1. Make sure you have everything commited in GIT with no pending changes.

2. Convert the MATLAB code using GPT5. Copy and paste in Cursor. This is the prompt "Convert this MATLAB function to Python. Do not type the inputs. Use keywords Python calling conventions. Follow the MATLAB code scrupulously, using the same variable names. If a custom sub-function is missing, put a comment but do not write the function. Assume EEG is a dictionary, not an object (EEG['chanlocs'] access channels and EEG['chanlocs'][0]['X'] accesses the coordinate X of the channels." 

3. Create the testcase, copying an existing one and telling GPT5 to use it. "Create a test case for this function using the template from another function below"

4. Comment the MATLAB test cases. Run and debug until the non-MATLAB test cases do not crash.

5. Uncomment one of the MATLAB test cases at a time and run. Use the prompt. "Please fix the issue "xxxxxxxxxxxx". You can find the Python environment in .venv folder if you need to run Python code. Remember that function @x.py has the equivalent @x.m which you can also modify to add comment that will be visible when you run the test cases."
   
6. Have 4 run automatically, then make a diff on the difference. Commit if relevant and revert if not. 

7. Iterate to fix all MATLAB cases. Make sure you have 90% code coverage.

## Manual debugging

4. Start the debugger in both MATLAB and Python and compare. Note that it is better to use the debugger on Python file than Jupyter Notebook (could not get it to stop)
