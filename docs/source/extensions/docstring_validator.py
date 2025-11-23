"""
Sphinx extension to validate docstrings and provide warnings for NumPy docstring format issues.

This extension intercepts autodoc events and validates docstrings against NumPy format.
"""

import re
from typing import Any, Dict
import warnings


def check_docstring_format(docstring: str, name: str) -> list:
    """
    Check docstring for common NumPy format issues.
    
    Parameters
    ----------
    docstring : str
        The docstring to check
    name : str
        Name of the object being checked
        
    Returns
    -------
    list
        List of issues found
    """
    issues = []
    
    if not docstring:
        return issues
    
    lines = docstring.split('\n')
    
    # Check 1: Section headers should be followed by a line of dashes
    section_headers = ['Parameters', 'Returns', 'Yields', 'Raises', 'Examples', 'Notes', 'References']
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if this is a section header
        if stripped in section_headers:
            # Next line should contain dashes
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if not next_line.strip() or not all(c == '-' for c in next_line.strip()):
                    issues.append(f"Section '{stripped}' at line {i+1} should be followed by a line of dashes")
            else:
                issues.append(f"Section '{stripped}' at line {i+1} is at end of docstring")
        
        # Check 2: Parameter descriptions should not be over-indented
        if i > 0:
            # Find if we're in Parameters section
            for j in range(max(0, i-10), i):
                if lines[j].strip() == 'Parameters':
                    # Check indentation of parameter lines
                    if ':' in stripped and not stripped.startswith('    '):
                        # Parameter definition should have standard indentation
                        if line.startswith('        '):  # 8 spaces is too much
                            issues.append(f"Parameter definition at line {i+1} has excessive indentation (should be 4 spaces)")
                    break
    
    # Check 3: Return section should have proper format
    in_returns = False
    for i, line in enumerate(lines):
        if line.strip() == 'Returns':
            in_returns = True
            # Check next 10 lines for proper format
            for j in range(i+2, min(i+12, len(lines))):
                l = lines[j]
                if l.strip() and not l.startswith('    '):
                    in_returns = False
                    break
                # Should have type name, then description
                if ':' in l and l.startswith('    ') and not l.startswith('        '):
                    # This looks like a return item, check if next line is description
                    if j + 1 < len(lines):
                        next_l = lines[j+1]
                        if next_l.strip() and not next_l.startswith('        '):
                            issues.append(f"Return description at line {j+2} should be indented more than return name")
    
    return issues


def process_docstring(app: Any, what: str, name: str, obj: Any, options: Dict, lines: list) -> None:
    """
    Process docstrings during sphinx autodoc.
    
    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application object
    what : str
        The type of object being documented
    name : str
        The fully qualified name of the object
    obj : Any
        The object being documented
    options : dict
        Options passed to the directive
    lines : list
        The docstring lines
    """
    if not lines:
        return
    
    docstring = '\n'.join(lines)
    issues = check_docstring_format(docstring, name)
    
    if issues:
        for issue in issues:
            warnings.warn(
                f"Docstring format issue in {name}: {issue}",
                UserWarning
            )


def setup(app: Any) -> Dict[str, Any]:
    """
    Setup the extension.
    
    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application
        
    Returns
    -------
    dict
        Extension metadata
    """
    app.connect('autodoc-process-docstring', process_docstring)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
