# EEGPrep Documentation Infrastructure - Quick Reference Guide

## Essential Components Summary

### 1. Core Dependencies (Phase 2)
```
sphinx>=7.0
pydata-sphinx-theme>=0.14.0
sphinx-gallery>=0.14.0
sphinx-autodoc-typehints>=1.25.0
numpydoc>=1.6.0
sphinx-design>=0.5.0
myst-parser>=1.0.0
sphinx-copybutton>=0.5.0
sphinx-togglebutton>=0.3.0
sphinxcontrib-spelling>=7.1.0
```

### 2. Directory Structure (Phase 1)
```
docs/
├── source/
│   ├── conf.py
│   ├── index.rst
│   ├── api/
│   ├── user_guide/
│   ├── examples/
│   ├── _static/
│   └── _templates/
├── build/
├── Makefile
└── .readthedocs.yml
```

### 3. Key Configuration Files

#### conf.py (Sphinx Configuration)
- Project metadata
- Extensions list
- Theme settings
- Autodoc options
- Sphinx-gallery config
- Intersphinx mappings

#### .readthedocs.yml (ReadTheDocs)
- Python version
- Build requirements
- Build command
- Output directory

#### Makefile (Build Automation)
- `make html` - Build HTML
- `make clean` - Clean build
- `make linkcheck` - Check links
- `make spelling` - Check spelling

### 4. API Documentation Categories

| Category | File | Functions |
|----------|------|-----------|
| Core | api/core.rst | bids_preproc, bids_list_eeg_files, eeg_checkset, EEGobj |
| Preprocessing | api/preprocessing.rst | clean_artifacts, clean_asr, clean_flatlines, clean_channels, clean_windows, clean_drifts, eeg_interp |
| ICA | api/ica.rst | eeg_picard, iclabel, ICL_feature_extractor |
| Signal Processing | api/signal_processing.rst | eeg_autocorr, eeg_autocorr_welch, eeg_rpsd, pop_resample, pop_rmbase |
| I/O | api/io.rst | pop_load_frombids, pop_loadset, pop_loadset_h5, pop_saveset, eeg_eeg2mne, eeg_mne2eeg |
| Utils | api/utils.rst | eeg_compare, eeg_decodechan, topoplot, utility modules |

### 5. User Guide Sections

| Section | File | Content |
|---------|------|---------|
| Overview | user_guide/index.rst | Guide overview and navigation |
| Installation | user_guide/installation.rst | Installation methods and requirements |
| Quick Start | user_guide/quickstart.rst | 5-minute quick start |
| Pipeline | user_guide/preprocessing_pipeline.rst | Pipeline overview and steps |
| BIDS | user_guide/bids_workflow.rst | BIDS workflow documentation |
| Configuration | user_guide/configuration.rst | Configuration options |
| Advanced | user_guide/advanced_topics.rst | Advanced topics |

### 6. Example Scripts

| Script | Purpose |
|--------|---------|
| plot_basic_preprocessing.py | Basic preprocessing workflow |
| plot_bids_pipeline.py | Full BIDS pipeline |
| plot_artifact_removal.py | Artifact removal demonstration |
| plot_ica_and_iclabel.py | ICA and ICLabel workflow |
| plot_channel_interpolation.py | Channel interpolation |
| plot_mne_integration.py | MNE-Python integration |

### 7. Supporting Documentation

| File | Purpose |
|------|---------|
| contributing.rst | Contribution guidelines |
| development.rst | Development setup |
| faq.rst | Frequently asked questions |
| references.rst | Publications and resources |
| changelog.rst | Version history |
| glossary.rst | EEG terminology |

### 8. Build and Deployment

#### Local Building
```bash
cd docs
make html          # Build HTML
make clean         # Clean build
make linkcheck      # Check links
make spelling      # Check spelling
make serve         # Serve locally
```

#### Deployment Targets
- **ReadTheDocs** (primary) - Automatic builds on push
- **GitHub Pages** (optional) - Manual deployment
- **Local server** - Development and testing

### 9. Theme Configuration (pydata-sphinx-theme)

Key settings in conf.py:
- Logo (light and dark versions)
- Color scheme
- Navigation structure
- Sidebar configuration
- Footer content
- Search settings
- Version switcher

### 10. Quality Assurance Checks

| Check | Tool | Purpose |
|-------|------|---------|
| Link validation | sphinx-linkcheck | Find broken links |
| Spell checking | sphinxcontrib-spelling | Find misspellings |
| Docstring validation | Custom script | Verify docstring format |
| Build warnings | Sphinx | Catch documentation issues |

### 11. Sphinx Extensions and Their Roles

| Extension | Role |
|-----------|------|
| autodoc | Extract API from docstrings |
| napoleon | Parse NumPy-style docstrings |
| sphinx-gallery | Generate example galleries |
| intersphinx | Link to external docs |
| sphinx-autodoc-typehints | Display type hints |
| myst-parser | Support Markdown |
| sphinx-design | Enhanced UI components |
| sphinx-copybutton | Copy button for code |
| sphinx-togglebutton | Collapsible sections |

### 12. Intersphinx Mappings

Link to external documentation:
- MNE-Python
- NumPy
- SciPy
- Matplotlib
- Python standard library

### 13. Implementation Phases

| Phase | Duration | Focus |
|-------|----------|-------|
| 1 | Week 1-2 | Sphinx setup and configuration |
| 2 | Week 1-2 | Dependencies and requirements |
| 3 | Week 3-4 | API documentation structure |
| 4 | Week 5-6 | User guides and tutorials |
| 5 | Week 7-8 | Example gallery |
| 6 | Week 9 | Supporting documentation |
| 7 | Week 10 | Build automation |
| 8 | Week 11 | Theme customization |
| 9 | Week 12 | Deployment configuration |
| 10 | Week 12 | Quality assurance |

### 14. Success Metrics

- [ ] Documentation builds without errors
- [ ] All public API functions documented
- [ ] At least 5 working example scripts
- [ ] User guide covers main workflows
- [ ] Theme matches MNE-Python quality
- [ ] Automated builds on every commit
- [ ] Link checking passes
- [ ] Spell checking passes
- [ ] Mobile-responsive design
- [ ] Search functionality works

### 15. Common Commands

```bash
# Install documentation dependencies
pip install eegprep[docs]

# Build documentation
cd docs && make html

# Clean build artifacts
cd docs && make clean

# Check for broken links
cd docs && make linkcheck

# Check spelling
cd docs && make spelling

# Serve documentation locally
cd docs && make serve

# View built documentation
open build/html/index.html
```

### 16. File Naming Conventions

- **RST files**: Use lowercase with underscores (e.g., `quickstart.rst`)
- **Python examples**: Use `plot_` prefix (e.g., `plot_basic_preprocessing.py`)
- **Static assets**: Use descriptive names (e.g., `logo.png`, `custom.css`)
- **API files**: Use module names (e.g., `preprocessing.rst`, `io.rst`)

### 17. Docstring Format (NumPy Style)

```python
def function_name(param1, param2):
    """
    Brief description.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
    
    Returns
    -------
    return_type
        Description of return value.
    
    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result)
    
    Notes
    -----
    Additional notes about the function.
    
    See Also
    --------
    related_function : Related function description.
    """
```

### 18. Cross-Referencing

```rst
# Link to function
:func:`eegprep.bids_preproc`

# Link to class
:class:`eegprep.EEGobj`

# Link to module
:mod:`eegprep.utils`

# Link to external docs
:meth:`numpy.ndarray.reshape`

# Link to section
:ref:`user-guide-installation`
```

### 19. ReadTheDocs Integration

- Connect GitHub repository
- Enable automatic builds
- Configure build environment
- Set up version management
- Configure webhooks
- Enable email notifications

### 20. GitHub Actions Workflow

Triggers:
- Push to main/develop
- Pull requests
- Manual trigger

Actions:
- Build documentation
- Run link checker
- Run spell checker
- Deploy to ReadTheDocs
- Comment on PRs

---

## Quick Start for Implementation

1. **Phase 1-2**: Set up Sphinx and install dependencies
2. **Phase 3**: Create API documentation structure
3. **Phase 4**: Write user guides
4. **Phase 5**: Create example scripts
5. **Phase 6**: Add supporting documentation
6. **Phase 7**: Set up build automation
7. **Phase 8**: Customize theme
8. **Phase 9**: Configure deployment
9. **Phase 10**: Implement QA checks

---

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/)
- [sphinx-gallery](https://sphinx-gallery.github.io/)
- [ReadTheDocs](https://readthedocs.org/)
- [MNE-Python Documentation](https://mne.tools/)

---

## Contact and Support

For questions about the documentation infrastructure:
- Check the DOCUMENTATION_INFRASTRUCTURE_PLAN.md for detailed information
- Review DOCUMENTATION_ARCHITECTURE_DIAGRAMS.md for visual architecture
- Consult the Sphinx documentation for specific questions
