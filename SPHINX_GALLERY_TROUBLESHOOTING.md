# Sphinx Gallery Configuration Troubleshooting Guide

## Overview
This document describes the issues encountered with the Sphinx Gallery configuration for the eegprep documentation and how they were resolved.

## Issues Found and Fixed

### 1. Invalid Configuration Parameter: `doc_pattern`

**Issue**: The `sphinx_gallery_conf` dictionary in `docs/source/conf.py` contained an invalid parameter `"doc_pattern": r"\.rst$"` on line 176.

**Error Message**:
```
sphinx.errors.ExtensionError: Example directory does not have a GALLERY_HEADER file...
```

**Root Cause**: The `doc_pattern` parameter is not a valid Sphinx Gallery configuration option. This parameter does not exist in the Sphinx Gallery API and was likely added based on incorrect documentation or assumptions.

**Solution**: Removed the invalid `"doc_pattern": r"\.rst$"` line from the `sphinx_gallery_conf` dictionary.

**Valid Sphinx Gallery Parameters** (for reference):
- `examples_dirs`: Directory containing example scripts
- `gallery_dirs`: Output directory for generated gallery
- `filename_pattern`: Pattern to match example filenames
- `ignore_pattern`: Pattern for files to ignore
- `plot_gallery`: Whether to execute examples
- `download_all_examples`: Whether to download all examples
- `abort_on_example_error`: Whether to abort on example errors
- `image_srcset`: Image srcset configuration
- `default_thumb_file`: Default thumbnail file
- `line_numbers`: Show line numbers in code blocks
- `remove_config_comments`: Remove config comments from code blocks
- `expected_failing_examples`: Set of expected failing examples
- `passing_examples`: List of passing examples
- `stale_examples`: List of stale examples
- `run_stale_examples`: Whether to run stale examples
- `backreferences_dir`: Directory for backreferences

### 2. Incorrect Examples Directory Path

**Issue**: The `examples_dirs` configuration was set to `"../examples"` which resolved to `docs/examples` instead of the actual location at `docs/source/examples`.

**Error Message**:
```
sphinx.errors.ExtensionError: Example directory /Users/baristim/Projects/eegprep/docs/source/../examples does not have a GALLERY_HEADER file...
```

**Root Cause**: The path was relative to `docs/source/` but pointed one level up, causing it to look in the wrong directory.

**Solution**: Changed `"examples_dirs": "../examples"` to `"examples_dirs": "examples"` to correctly reference the examples directory at `docs/source/examples/`.

### 3. Missing GALLERY_HEADER File

**Issue**: Sphinx Gallery requires a GALLERY_HEADER file in the examples directory to introduce the gallery.

**Error Message**:
```
sphinx.errors.ExtensionError: Example directory ... does not have a GALLERY_HEADER file with one of the expected file extensions ['.txt', '.md', '.rst'].
```

**Root Cause**: The examples directory at `docs/source/examples/` did not contain a GALLERY_HEADER file.

**Solution**: Created `docs/source/examples/README.txt` with appropriate gallery introduction content:
```
================
Example Gallery
================

This gallery contains example scripts demonstrating various features and workflows of eegprep.

Each example shows how to use different components of the eegprep package for EEG preprocessing tasks.
```

## Configuration Changes Summary

### File: `docs/source/conf.py`

**Changes Made**:
1. Removed invalid parameter: `"doc_pattern": r"\.rst$"`
2. Fixed examples directory path: `"../examples"` → `"examples"`

**Before**:
```python
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    # ... other config ...
    "backreferences_dir": None,
    "doc_pattern": r"\.rst$",  # INVALID - REMOVED
}
```

**After**:
```python
sphinx_gallery_conf = {
    "examples_dirs": "examples",
    # ... other config ...
    "backreferences_dir": None,
}
```

### File: `docs/source/examples/README.txt` (Created)

Created a new GALLERY_HEADER file to introduce the example gallery.

## Build Results

**Final Build Status**: ✅ **SUCCESS**

The documentation built successfully with the following output:
```
Sphinx-Gallery successfully executed 0 out of 5 files subselected by:
    gallery_conf["filename_pattern"] = '/plot_'
    gallery_conf["ignore_pattern"]   = '__init__\\.py'
```

**Build Output Location**: `/Users/baristim/Projects/eegprep/docs/build/html/`

**Generated Pages**:
- API documentation (core, ica, io, preprocessing, signal_processing, utils)
- Auto-generated examples gallery
- User guide sections
- Contributing and development guides
- FAQ, glossary, and references

## Verification Steps

To verify the documentation builds correctly:

```bash
cd /Users/baristim/Projects/eegprep/docs
conda run -n eegprep make clean
conda run -n eegprep make html
```

Expected output should show:
- No ConfigError or ExtensionError
- Successful Sphinx-Gallery execution
- HTML files generated in `build/html/`

## Prevention Tips

1. **Always validate configuration parameters** against the official Sphinx Gallery documentation
2. **Use relative paths carefully** - ensure they resolve correctly from the configuration file location
3. **Create required GALLERY_HEADER files** when using Sphinx Gallery
4. **Test builds locally** before committing configuration changes
5. **Check Sphinx Gallery version compatibility** - some parameters may vary between versions

## References

- Sphinx Gallery Documentation: https://sphinx-gallery.github.io/
- Sphinx Gallery Configuration: https://sphinx-gallery.github.io/stable/configuration.html
- eegprep Documentation: `/Users/baristim/Projects/eegprep/docs/`

## Build Environment

- **Conda Environment**: eegprep
- **Sphinx Version**: 8.2.3
- **Python Version**: 3.13.0
- **Platform**: macOS Sequoia (arm64)
