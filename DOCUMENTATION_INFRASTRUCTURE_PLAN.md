# EEGPrep Documentation Infrastructure Plan

## Executive Summary

This document outlines a comprehensive documentation infrastructure for the eegprep project using Sphinx with the pydata-sphinx-theme. The plan is modeled after professional-grade documentation systems like MNE-Python and includes all necessary components for a production-ready documentation system.

**Project Context:**
- Package: eegprep v0.2.23
- Purpose: Python EEG preprocessing pipeline with MATLAB-to-Python equivalence
- Target Audience: Neuroscientists, EEG researchers, developers
- Status: Pre-release (planned release end of 2025)

---

## Architecture Overview

```
Documentation System Architecture:

┌─────────────────────────────────────────────────────────────────┐
│                    Documentation Infrastructure                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Source Files (docs/source/)                             │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  • conf.py (Sphinx configuration)                        │   │
│  │  • index.rst (main entry point)                          │   │
│  │  • api/ (API reference - auto-generated)                 │   │
│  │  • user_guide/ (tutorials and guides)                    │   │
│  │  • examples/ (gallery examples)                          │   │
│  │  • _static/ (CSS, images, assets)                        │   │
│  │  • _templates/ (custom HTML templates)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Sphinx Build System                                     │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Extensions:                                             │   │
│  │  • autodoc (API extraction from docstrings)              │   │
│  │  • napoleon (NumPy/Google docstring parsing)             │   │
│  │  • sphinx-gallery (example gallery generation)           │   │
│  │  • intersphinx (cross-project linking)                   │   │
│  │  • sphinx-autodoc-typehints (type hints)                 │   │
│  │  • myst-parser (Markdown support)                        │   │
│  │  • sphinx-design (UI components)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Theme: pydata-sphinx-theme                              │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  • Responsive design                                     │   │
│  │  • Dark mode support                                     │   │
│  │  • Integrated search                                     │   │
│  │  • Navigation sidebar                                    │   │
│  │  • Mobile-friendly                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Build Output (docs/build/html/)                         │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  • Static HTML files                                     │   │
│  │  • Search index                                          │   │
│  │  • API documentation                                     │   │
│  │  • Gallery examples                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Deployment Targets                                      │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  • ReadTheDocs (primary)                                 │   │
│  │  • GitHub Pages (optional)                               │   │
│  │  • Local development server                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Sphinx Configuration

### Objectives
- Establish Sphinx project structure
- Configure pydata-sphinx-theme
- Set up essential extensions
- Create main documentation entry point

### Deliverables

#### 1.1 Directory Structure
```
docs/
├── source/
│   ├── conf.py                 # Sphinx configuration
│   ├── index.rst               # Main documentation page
│   ├── api/                    # API reference
│   ├── user_guide/             # User guides and tutorials
│   ├── examples/               # Example scripts for gallery
│   ├── _static/                # Static assets (CSS, images)
│   ├── _templates/             # Custom HTML templates
│   └── _build/                 # Build output (generated)
├── build/                      # Build artifacts
├── Makefile                    # Build automation
└── .readthedocs.yml            # ReadTheDocs configuration
```

#### 1.2 conf.py Configuration
Key sections:
- Project metadata (name, version, author)
- Extensions list
- Theme configuration
- HTML output settings
- Autodoc options
- Sphinx-gallery configuration
- Intersphinx mappings (MNE, NumPy, SciPy, Matplotlib)

#### 1.3 index.rst Structure
- Welcome section with project overview
- Quick links to key sections
- Feature highlights
- Installation quick start
- Table of contents (toctree)

---

## Phase 2: Documentation Dependencies

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| sphinx | >=7.0 | Core documentation generator |
| pydata-sphinx-theme | >=0.14.0 | Modern, responsive theme |
| sphinx-gallery | >=0.14.0 | Auto-generate example galleries |
| sphinx-autodoc-typehints | >=1.25.0 | Type hint documentation |
| numpydoc | >=1.6.0 | NumPy-style docstring parsing |
| sphinx-design | >=0.5.0 | Enhanced UI components |
| myst-parser | >=1.0.0 | Markdown support |
| sphinx-copybutton | >=0.5.0 | Copy button for code blocks |
| sphinx-togglebutton | >=0.3.0 | Collapsible sections |
| linkify-it-py | >=2.0.0 | Automatic link detection |
| sphinxcontrib-spelling | >=7.1.0 | Spell checking |
| sphinx-linkcheck | >=1.0.0 | Link validation |

### Implementation
- Add `[docs]` optional dependency group to pyproject.toml
- Create requirements-docs.txt for easy installation
- Document installation: `pip install eegprep[docs]`

---

## Phase 3: API Documentation Structure

### Objectives
- Auto-generate API reference from docstrings
- Organize API by functional categories
- Ensure comprehensive coverage of public API

### API Categories

#### 3.1 Core Functions (api/core.rst)
- `bids_preproc()` - Main preprocessing pipeline
- `bids_list_eeg_files()` - BIDS file discovery
- `eeg_checkset()` - Data validation
- `EEGobj` - Object-oriented interface

#### 3.2 Preprocessing Functions (api/preprocessing.rst)
- `clean_artifacts()` - Artifact removal
- `clean_asr()` - Artifact Subspace Reconstruction
- `clean_flatlines()` - Flatline detection
- `clean_channels()` - Bad channel removal
- `clean_windows()` - Window rejection
- `clean_drifts()` - Drift correction
- `eeg_interp()` - Channel interpolation

#### 3.3 ICA and Component Analysis (api/ica.rst)
- `eeg_picard()` - Picard ICA
- `iclabel()` - ICLabel classification
- `ICL_feature_extractor()` - Feature extraction

#### 3.4 Signal Processing (api/signal_processing.rst)
- `eeg_autocorr()` - Autocorrelation
- `eeg_autocorr_welch()` - Welch autocorrelation
- `eeg_rpsd()` - Relative power spectral density
- `pop_resample()` - Resampling
- `pop_rmbase()` - Baseline removal

#### 3.5 I/O Functions (api/io.rst)
- `pop_load_frombids()` - Load from BIDS
- `pop_loadset()` - Load EEGLAB format
- `pop_loadset_h5()` - Load HDF5 format
- `pop_saveset()` - Save EEGLAB format
- `eeg_eeg2mne()` - Convert to MNE
- `eeg_mne2eeg()` - Convert from MNE

#### 3.6 Utilities (api/utils.rst)
- `eeg_compare()` - Data comparison
- `eeg_decodechan()` - Channel decoding
- `topoplot()` - Topographic plotting
- Utility modules (coords, spatial, stats, etc.)

### Implementation Details
- Use `automodule` directive with `:members:` option
- Configure autodoc to extract docstrings automatically
- Set up intersphinx for cross-referencing with MNE, NumPy, SciPy
- Generate API documentation on every build

---

## Phase 4: User Guide and Tutorials

### Objectives
- Provide comprehensive user documentation
- Create step-by-step tutorials
- Document common workflows

### Documentation Files

#### 4.1 user_guide/index.rst
- Overview of user guide sections
- Quick navigation
- Learning path recommendations

#### 4.2 user_guide/installation.rst
- Installation methods (pip, conda, from source)
- System requirements
- Optional dependencies
- Troubleshooting

#### 4.3 user_guide/quickstart.rst
- 5-minute quick start
- Basic preprocessing example
- Loading and saving data
- Visualization

#### 4.4 user_guide/preprocessing_pipeline.rst
- Pipeline overview
- Step-by-step explanation
- Parameter tuning
- Quality control

#### 4.5 user_guide/bids_workflow.rst
- BIDS dataset structure
- Loading BIDS data
- Running batch preprocessing
- Output structure
- Integration with other tools

#### 4.6 user_guide/configuration.rst
- Configuration options
- EEG_OPTIONS dataclass
- Customizing pipeline parameters
- Advanced settings

#### 4.7 user_guide/advanced_topics.rst
- Custom preprocessing chains
- Extending the pipeline
- Integration with MNE-Python
- Parallel processing

---

## Phase 5: Gallery Examples

### Objectives
- Provide executable example scripts
- Auto-generate gallery with output
- Demonstrate common use cases

### Example Scripts

#### 5.1 examples/plot_basic_preprocessing.py
- Load sample EEG data
- Run basic preprocessing
- Visualize results
- Save output

#### 5.2 examples/plot_bids_pipeline.py
- Load BIDS dataset
- Run full preprocessing pipeline
- Generate reports
- Export derivatives

#### 5.3 examples/plot_artifact_removal.py
- Demonstrate artifact detection
- Compare different methods
- Visualize before/after
- Parameter sensitivity

#### 5.4 examples/plot_ica_and_iclabel.py
- Perform ICA decomposition
- Run ICLabel classification
- Visualize components
- Remove artifacts

#### 5.5 examples/plot_channel_interpolation.py
- Identify bad channels
- Interpolate channels
- Verify interpolation quality
- Visualize topography

#### 5.6 examples/plot_mne_integration.py
- Convert to MNE format
- Use MNE tools
- Convert back to EEGLAB
- Compare results

### Implementation
- Use sphinx-gallery to auto-generate gallery
- Include output plots and console output
- Generate thumbnail images
- Create gallery index page

---

## Phase 6: Additional Documentation

### Objectives
- Provide supporting documentation
- Establish contribution guidelines
- Document development workflow

### Documentation Files

#### 6.1 contributing.rst
- How to contribute
- Code style guidelines
- Testing requirements
- Documentation standards
- Pull request process

#### 6.2 development.rst
- Development environment setup
- Running tests
- Building documentation locally
- Debugging tips
- Release process

#### 6.3 faq.rst
- Common questions
- Troubleshooting
- Performance tips
- Comparison with MATLAB EEGLAB

#### 6.4 references.rst
- Key publications
- Related tools
- External resources
- Citation information

#### 6.5 changelog.rst
- Link to GitHub releases
- Version history
- Breaking changes
- Migration guides

#### 6.6 glossary.rst
- EEG terminology
- Technical terms
- Acronyms
- Cross-references

---

## Phase 7: Build Configuration and Automation

### Objectives
- Automate documentation building
- Enable CI/CD integration
- Configure deployment

### Deliverables

#### 7.1 Makefile
```makefile
# Key targets:
make html          # Build HTML documentation
make clean         # Clean build artifacts
make linkcheck      # Check for broken links
make spelling      # Check spelling
make serve         # Serve documentation locally
```

#### 7.2 .readthedocs.yml
- Python version specification
- Build requirements
- Build command
- Output directory
- Versioning strategy
- Webhook configuration

#### 7.3 GitHub Actions Workflow (.github/workflows/docs.yml)
- Trigger on push to main/develop
- Build documentation
- Run link checker
- Run spell checker
- Deploy to GitHub Pages (optional)
- Comment on PRs with build status

#### 7.4 Automated API Documentation
- Configure autodoc to auto-discover modules
- Generate API stubs on build
- Update API reference automatically
- Validate docstring format

#### 7.5 HTML Output Configuration
- Logo and favicon
- Sidebar configuration
- Navigation structure
- Search settings
- Analytics integration (optional)

---

## Phase 8: Theme and Styling

### Objectives
- Customize pydata-sphinx-theme
- Maintain visual consistency
- Enhance user experience

### Deliverables

#### 8.1 _static/custom.css
- Custom color scheme
- Font customization
- Layout adjustments
- Responsive design tweaks
- Dark mode adjustments

#### 8.2 Theme Configuration (conf.py)
```python
html_theme_options = {
    "logo": {
        "image_light": "logo.png",
        "image_dark": "logo-dark.png",
    },
    "navbar_align": "left",
    "navbar_persistent": "",
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc"],
    "header_links_before_dropdown": 4,
    "switcher": {
        "json_url": "https://eegprep.readthedocs.io/en/_static/switcher.json",
        "version_match": "latest",
    },
    "check_switcher": False,
    "announcement": "",
}
```

#### 8.3 Assets
- Project logo (light and dark versions)
- Favicon
- Custom icons
- Banner images

#### 8.4 Navigation Structure
- Main navigation menu
- Sidebar organization
- Breadcrumb navigation
- Related links

---

## Phase 9: Deployment Configuration

### Objectives
- Configure production deployment
- Set up versioning strategy
- Enable multiple deployment targets

### Deliverables

#### 9.1 ReadTheDocs Configuration
- Project setup on ReadTheDocs
- Build environment configuration
- Webhook integration with GitHub
- Email notifications
- Version management

#### 9.2 GitHub Pages Deployment (Optional)
- GitHub Actions workflow for deployment
- Custom domain configuration
- CNAME file setup
- Branch protection rules

#### 9.3 Deployment Documentation
- Deployment process
- Rollback procedures
- Monitoring and alerts
- Troubleshooting

#### 9.4 Versioning Strategy
- Documentation for each release
- Latest version pointer
- Stable version designation
- Development version
- Version switcher configuration

#### 9.5 Redirects and Deprecations
- URL redirect rules
- Deprecated page handling
- Version-specific documentation
- Migration guides

---

## Phase 10: Quality Assurance

### Objectives
- Ensure documentation quality
- Automate quality checks
- Establish standards

### Deliverables

#### 10.1 Docstring Validation
- Validate docstring format
- Check for missing documentation
- Verify parameter documentation
- Validate return type documentation
- CI/CD integration

#### 10.2 Documentation Style Guide
- Writing style guidelines
- Code example standards
- Formatting conventions
- Terminology consistency
- Tone and voice guidelines

#### 10.3 Link Checking
- Automated link validation
- Broken link detection
- External link verification
- CI/CD integration

#### 10.4 Spell Checking
- Spell checker configuration
- Custom dictionary
- Technical term handling
- CI/CD integration

#### 10.5 Documentation Review Checklist
- Content completeness
- Accuracy verification
- Example functionality
- Link validity
- Formatting consistency
- Accessibility compliance

---

## Implementation Timeline

### Week 1-2: Foundation (Phases 1-2)
- Set up Sphinx project structure
- Configure pydata-sphinx-theme
- Add documentation dependencies
- Create main index.rst

### Week 3-4: API Documentation (Phase 3)
- Create API reference structure
- Configure autodoc
- Generate API documentation
- Set up intersphinx

### Week 5-6: User Guide (Phase 4)
- Write installation guide
- Create quickstart tutorial
- Document preprocessing pipeline
- Write BIDS workflow guide

### Week 7-8: Examples and Gallery (Phase 5)
- Create example scripts
- Configure sphinx-gallery
- Generate gallery
- Add example documentation

### Week 9: Supporting Documentation (Phase 6)
- Write contributing guide
- Create development guide
- Write FAQ
- Add references and glossary

### Week 10: Build Automation (Phase 7)
- Create Makefile
- Configure ReadTheDocs
- Set up GitHub Actions
- Configure deployment

### Week 11: Styling and Customization (Phase 8)
- Customize CSS
- Add project assets
- Configure theme options
- Test responsive design

### Week 12: Deployment and QA (Phases 9-10)
- Deploy to ReadTheDocs
- Set up versioning
- Implement QA checks
- Final review and testing

---

## Technology Stack

### Core Tools
- **Sphinx 7.0+**: Documentation generator
- **pydata-sphinx-theme 0.14.0+**: Modern responsive theme
- **Python 3.10+**: Build environment

### Extensions
- **sphinx-autodoc-typehints**: Type hint support
- **sphinx-gallery**: Example gallery generation
- **numpydoc**: NumPy docstring parsing
- **myst-parser**: Markdown support
- **sphinx-design**: UI components
- **sphinx-copybutton**: Code copy functionality
- **sphinx-togglebutton**: Collapsible sections

### Build and Deployment
- **ReadTheDocs**: Primary hosting
- **GitHub Actions**: CI/CD automation
- **Makefile**: Local build automation

### Quality Assurance
- **sphinx-linkcheck**: Link validation
- **sphinxcontrib-spelling**: Spell checking
- **pytest**: Documentation testing

---

## Key Features

### 1. Comprehensive API Documentation
- Auto-generated from docstrings
- Type hints included
- Cross-references to related functions
- Examples for each function

### 2. Professional User Guide
- Installation instructions
- Quick start tutorial
- Detailed workflow documentation
- Advanced topics

### 3. Executable Examples
- Gallery of example scripts
- Auto-generated output
- Thumbnail previews
- Downloadable notebooks

### 4. Modern Theme
- Responsive design
- Dark mode support
- Integrated search
- Mobile-friendly

### 5. Automated Quality Checks
- Link validation
- Spell checking
- Docstring validation
- CI/CD integration

### 6. Multiple Deployment Options
- ReadTheDocs (primary)
- GitHub Pages (optional)
- Local development server

---

## Success Criteria

1. ✓ Documentation builds without errors
2. ✓ All public API functions documented
3. ✓ At least 5 working example scripts
4. ✓ User guide covers main workflows
5. ✓ Theme matches MNE-Python quality
6. ✓ Automated builds on every commit
7. ✓ Link checking passes
8. ✓ Spell checking passes
9. ✓ Mobile-responsive design
10. ✓ Search functionality works

---

## Maintenance and Updates

### Regular Tasks
- Update documentation with new features
- Review and update examples
- Fix broken links
- Update dependencies
- Review and improve user feedback

### Quarterly Reviews
- Check for outdated information
- Update screenshots and diagrams
- Review user feedback
- Assess documentation coverage

### Annual Reviews
- Major documentation restructuring if needed
- Theme and styling updates
- Technology stack evaluation
- Accessibility audit

---

## References

### Similar Projects
- [MNE-Python Documentation](https://mne.tools/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

### Tools and Resources
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/)
- [sphinx-gallery](https://sphinx-gallery.github.io/)
- [ReadTheDocs](https://readthedocs.org/)

---

## Appendix: Configuration Templates

### A. pyproject.toml Dependencies
```toml
[project.optional-dependencies]
docs = [
    "sphinx>=7.0",
    "pydata-sphinx-theme>=0.14.0",
    "sphinx-gallery>=0.14.0",
    "sphinx-autodoc-typehints>=1.25.0",
    "numpydoc>=1.6.0",
    "sphinx-design>=0.5.0",
    "myst-parser>=1.0.0",
    "sphinx-copybutton>=0.5.0",
    "sphinx-togglebutton>=0.3.0",
    "sphinxcontrib-spelling>=7.1.0",
]
```

### B. ReadTheDocs Configuration (.readthedocs.yml)
```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
python:
  version: 3.11
  install:
    - method: pip
      path: .
      extra: docs
sphinx:
  configuration: docs/source/conf.py
formats:
  - pdf
  - epub
```

### C. GitHub Actions Workflow
```yaml
name: Documentation Build
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[docs]"
      - run: cd docs && make html
      - run: cd docs && make linkcheck
```

---

## Document Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-22 | Architecture Team | Initial comprehensive plan |

---

**Status**: Ready for Implementation
**Next Step**: Switch to Code mode to begin Phase 1 implementation
