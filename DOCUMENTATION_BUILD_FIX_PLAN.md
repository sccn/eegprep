# EEGPREP DOCUMENTATION BUILD FIX PLAN

## OBJECTIVE

Resolve all documentation build errors and warnings to achieve a clean, error-free build. The issues span three categories: example execution failures, source code docstring formatting errors, and RST documentation formatting issues.

---

## IDENTIFIED ISSUES

### CATEGORY 1: EXAMPLE EXECUTION FAILURES (4 examples)

#### Issue 1.1: clean_artifacts() doesn't accept sfreq parameter

**Files Affected:**
- `docs/source/examples/plot_artifact_removal.py` (line 135)
- `docs/source/examples/plot_ica_and_iclabel.py` (line 140)

**Problem:** Examples call `clean_artifacts(data, sfreq=sfreq, verbose=False)` but function doesn't accept `sfreq` parameter

**Solution:** Remove `sfreq` parameter from function calls

#### Issue 1.2: Montage channel mismatch

**Files Affected:**
- `docs/source/examples/plot_basic_preprocessing.py` (line 94)
- `docs/source/examples/plot_channel_interpolation.py` (line 235)

**Problem:** 7 channels missing from montage: `['Fc1', 'Fc2', 'Cp1', 'Cp2', 'Fc5', 'Fc6', 'Cp5']`

**Solution:** Use `on_missing='ignore'` parameter in `set_montage()` call

---

### CATEGORY 2: SOURCE CODE DOCSTRING FORMATTING ERRORS (14 files)

#### Issue 2.1: Unexpected indentation in docstrings

**Files Affected:**
- `src/eegprep/bids_preproc.py` (multiple lines)
- `src/eegprep/clean_artifacts.py` (line 87)
- `src/eegprep/eeg_decodechan.py` (line 15)
- `src/eegprep/eeg_interp.py` (line 12)
- `src/eegprep/eeg_lat2point.py` (lines 6, 22)
- `src/eegprep/eeg_point2lat.py` (lines 6, 18)
- `src/eegprep/pop_epoch.py` (line 15)
- `src/eegprep/pop_load_frombids.py` (lines 19, 31)
- `src/eegprep/pop_select.py` (line 3)

**Problem:** Docstrings have improper indentation causing reStructuredText parsing errors

**Solution:** Fix indentation in docstrings to follow proper RST format

---

### CATEGORY 3: RST DOCUMENTATION FORMATTING ISSUES (4 files)

#### Issue 3.1: Title overline/underline length mismatch

**Files Affected:**
- `docs/source/auto_examples/index.rst` (line 3): Title overline too short
- `docs/source/faq.rst` (line 77, 262): Title underline too short
- `docs/source/references.rst` (line 3): Title overline too short
- `docs/source/user_guide/preprocessing_pipeline.rst` (line 3): Title overline too short

**Problem:** RST requires title underlines/overlines to match title length

**Solution:** Adjust underline/overline length to match title

---

## RESOLUTION PLAN

### PHASE 1: FIX EXAMPLE EXECUTION ISSUES (4 files)

#### Step 1.1: Fix plot_artifact_removal.py
- Remove `sfreq=sfreq` parameter from `clean_artifacts()` call (line 135)
- **BEFORE:**
  ```python
  cleaned_artifacts = eegprep.clean_artifacts(
      data.copy(),
      sfreq=sfreq,
      verbose=False
  )
  ```
- **AFTER:**
  ```python
  cleaned_artifacts = eegprep.clean_artifacts(
      data.copy(),
      verbose=False
  )
  ```

#### Step 1.2: Fix plot_ica_and_iclabel.py
- Remove `sfreq=sfreq` parameter from `clean_artifacts()` call (line 140)
- **BEFORE:**
  ```python
  data_prep = eegprep.clean_artifacts(
      data.copy(),
      sfreq=sfreq,
      verbose=False
  )
  ```
- **AFTER:**
  ```python
  data_prep = eegprep.clean_artifacts(
      data.copy(),
      verbose=False
  )
  ```

#### Step 1.3: Fix plot_basic_preprocessing.py
- Add `on_missing='ignore'` parameter to `set_montage()` call (line 94)
- **BEFORE:** `info.set_montage(montage)`
- **AFTER:** `info.set_montage(montage, on_missing='ignore')`

#### Step 1.4: Fix plot_channel_interpolation.py
- Add `on_missing='ignore'` parameter to `set_montage()` call (line 235)
- **BEFORE:** `info.set_montage(montage)`
- **AFTER:** `info.set_montage(montage, on_missing='ignore')`

---

### PHASE 2: FIX SOURCE CODE DOCSTRING FORMATTING (14 files)

For each file listed below, fix docstring indentation errors:
- `src/eegprep/bids_preproc.py`
- `src/eegprep/clean_artifacts.py`
- `src/eegprep/eeg_decodechan.py`
- `src/eegprep/eeg_interp.py`
- `src/eegprep/eeg_lat2point.py`
- `src/eegprep/eeg_point2lat.py`
- `src/eegprep/pop_epoch.py`
- `src/eegprep/pop_load_frombids.py`
- `src/eegprep/pop_select.py`

**Common fixes needed:**
- Ensure proper RST formatting with correct indentation
- Fix block quote formatting (add blank lines before/after)
- Fix bullet list formatting (add blank lines before/after)
- Fix definition list formatting (add blank lines before/after)
- Ensure code examples are properly indented

---

### PHASE 3: FIX RST DOCUMENTATION FORMATTING (4 files)

#### Step 3.1: Fix docs/source/auto_examples/index.rst
- Adjust title overline to match title length (line 3)
- **BEFORE:**
  ```
  ==============
  Example Gallery
  ==============
  ```
- **AFTER:**
  ```
  ===============
  Example Gallery
  ===============
  ```

#### Step 3.2: Fix docs/source/faq.rst
- Adjust title underlines to match title lengths (lines 77, 262)
- Example: "How do I load EEG data?" needs 23 dashes, not 22

#### Step 3.3: Fix docs/source/references.rst
- Adjust title overline to match title length (line 3)

#### Step 3.4: Fix docs/source/user_guide/preprocessing_pipeline.rst
- Adjust title overline to match title length (line 3)

---

### PHASE 4: BUILD AND VALIDATE

#### Step 4.1: Run make clean
- Command: `cd /Users/baristim/Projects/eegprep/docs && conda run -n eegprep make clean`
- Purpose: Remove all build artifacts

#### Step 4.2: Run make html
- Command: `cd /Users/baristim/Projects/eegprep/docs && conda run -n eegprep make html`
- Purpose: Build documentation

#### Step 4.3: Verify build completes successfully
- Check for zero errors
- Check for zero warnings (or only acceptable warnings)
- Verify all examples are processed
- Verify all pages are generated
- Verify HTML output in `docs/build/html/`

---

## EXECUTION WORKFLOW

```
Phase 1: Fix Examples (4 files)
   |
   v
Phase 2: Fix Docstrings (14 files)
   |
   v
Phase 3: Fix RST Formatting (4 files)
   |
   v
Phase 4: Build & Validate
   ├─ make clean
   ├─ make html
   └─ Verify success
```

---

## ISSUE SUMMARY

| Category | Count | Severity | Impact |
|----------|-------|----------|--------|
| Example Execution Failures | 4 | Critical | Build fails |
| Docstring Formatting Errors | 14 | High | Build warnings |
| RST Formatting Issues | 4 | Medium | Build warnings |
| **TOTAL** | **22** | - | - |

---

## NEXT STEPS

1. Fix all 4 example files (Phase 1)
2. Fix all 14 source code docstring files (Phase 2)
3. Fix all 4 RST documentation files (Phase 3)
4. Run `make clean` and `make html` (Phase 4)
5. Validate the build completes successfully
6. Provide detailed completion report

This plan will result in a clean, error-free documentation build ready for GitHub Pages deployment.
