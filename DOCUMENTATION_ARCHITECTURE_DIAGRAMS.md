# EEGPrep Documentation Architecture Diagrams

## 1. Documentation Build Pipeline

```
Source Files
    ↓
    ├─ .rst files (reStructuredText)
    ├─ .md files (Markdown via myst-parser)
    ├─ Python docstrings (via autodoc)
    └─ Example scripts (via sphinx-gallery)
    ↓
Sphinx Processing
    ├─ Parse source files
    ├─ Extract docstrings
    ├─ Generate gallery examples
    ├─ Cross-reference linking
    └─ Build search index
    ↓
Theme Application (pydata-sphinx-theme)
    ├─ Apply HTML templates
    ├─ Inject CSS styling
    ├─ Add navigation
    └─ Configure responsive layout
    ↓
Quality Checks
    ├─ Link validation
    ├─ Spell checking
    ├─ Docstring validation
    └─ Build warnings
    ↓
Output
    ├─ HTML files
    ├─ Search index
    ├─ Static assets
    └─ PDF/EPUB (optional)
    ↓
Deployment
    ├─ ReadTheDocs
    ├─ GitHub Pages
    └─ Local server
```

## 2. Directory Structure

```
eegprep/
├── docs/
│   ├── source/
│   │   ├── conf.py                    # Sphinx configuration
│   │   ├── index.rst                  # Main page
│   │   ├── api/
│   │   │   ├── index.rst              # API overview
│   │   │   ├── core.rst               # Core functions
│   │   │   ├── preprocessing.rst      # Preprocessing functions
│   │   │   ├── ica.rst                # ICA functions
│   │   │   ├── signal_processing.rst  # Signal processing
│   │   │   ├── io.rst                 # I/O functions
│   │   │   └── utils.rst              # Utilities
│   │   ├── user_guide/
│   │   │   ├── index.rst              # User guide overview
│   │   │   ├── installation.rst       # Installation
│   │   │   ├── quickstart.rst         # Quick start
│   │   │   ├── preprocessing_pipeline.rst
│   │   │   ├── bids_workflow.rst      # BIDS workflow
│   │   │   ├── configuration.rst      # Configuration
│   │   │   └── advanced_topics.rst    # Advanced topics
│   │   ├── examples/
│   │   │   ├── plot_basic_preprocessing.py
│   │   │   ├── plot_bids_pipeline.py
│   │   │   ├── plot_artifact_removal.py
│   │   │   ├── plot_ica_and_iclabel.py
│   │   │   ├── plot_channel_interpolation.py
│   │   │   └── plot_mne_integration.py
│   │   ├── contributing.rst           # Contributing guide
│   │   ├── development.rst            # Development guide
│   │   ├── faq.rst                    # FAQ
│   │   ├── references.rst             # References
│   │   ├── changelog.rst              # Changelog
│   │   ├── glossary.rst               # Glossary
│   │   ├── _static/
│   │   │   ├── custom.css             # Custom styling
│   │   │   ├── logo.png               # Light logo
│   │   │   ├── logo-dark.png          # Dark logo
│   │   │   └── favicon.ico            # Favicon
│   │   └── _templates/
│   │       └── custom_theme.html      # Custom templates
│   ├── build/                         # Build output (generated)
│   │   └── html/
│   │       ├── index.html
│   │       ├── api/
│   │       ├── user_guide/
│   │       ├── examples/
│   │       ├── _static/
│   │       └── searchindex.js
│   ├── Makefile                       # Build automation
│   └── .readthedocs.yml               # ReadTheDocs config
├── .github/
│   └── workflows/
│       └── docs.yml                   # GitHub Actions workflow
├── pyproject.toml                     # Project config with docs dependencies
├── requirements-docs.txt              # Documentation dependencies
└── DOCUMENTATION_INFRASTRUCTURE_PLAN.md
```

## 3. Sphinx Extension Architecture

```
Sphinx Core
    ├─ autodoc
    │   ├─ Extracts docstrings from Python modules
    │   ├─ Generates API documentation
    │   └─ Supports type hints
    │
    ├─ napoleon
    │   ├─ Parses NumPy-style docstrings
    │   ├─ Parses Google-style docstrings
    │   └─ Converts to reStructuredText
    │
    ├─ sphinx-gallery
    │   ├─ Executes Python example scripts
    │   ├─ Captures output and plots
    │   ├─ Generates gallery HTML
    │   └─ Creates downloadable notebooks
    │
    ├─ intersphinx
    │   ├─ Links to external documentation
    │   ├─ MNE-Python
    │   ├─ NumPy
    │   ├─ SciPy
    │   └─ Matplotlib
    │
    ├─ sphinx-autodoc-typehints
    │   ├─ Extracts type hints
    │   ├─ Displays in documentation
    │   └─ Improves API clarity
    │
    ├─ myst-parser
    │   ├─ Parses Markdown files
    │   ├─ Converts to reStructuredText
    │   └─ Supports MyST extensions
    │
    ├─ sphinx-design
    │   ├─ Grid layouts
    │   ├─ Cards and tabs
    │   ├─ Dropdowns
    │   └─ Enhanced UI components
    │
    ├─ sphinx-copybutton
    │   └─ Copy button for code blocks
    │
    └─ sphinx-togglebutton
        └─ Collapsible sections
```

## 4. API Documentation Generation Flow

```
Python Source Code (src/eegprep/)
    ↓
Docstrings (NumPy style)
    ├─ Function/class description
    ├─ Parameters section
    ├─ Returns section
    ├─ Examples section
    └─ Notes section
    ↓
autodoc Extension
    ├─ Discovers modules
    ├─ Extracts docstrings
    ├─ Parses type hints
    └─ Generates .rst files
    ↓
napoleon Extension
    ├─ Parses NumPy docstring format
    ├─ Converts to reStructuredText
    └─ Formats sections
    ↓
sphinx-autodoc-typehints
    ├─ Extracts type annotations
    ├─ Displays in documentation
    └─ Creates cross-references
    ↓
API Reference Pages
    ├─ api/core.rst
    ├─ api/preprocessing.rst
    ├─ api/ica.rst
    ├─ api/signal_processing.rst
    ├─ api/io.rst
    └─ api/utils.rst
    ↓
HTML Output
    ├─ Function signatures
    ├─ Parameter descriptions
    ├─ Return type information
    ├─ Example code
    └─ Cross-references
```

## 5. Gallery Example Processing

```
Example Scripts (docs/source/examples/)
    ├─ plot_basic_preprocessing.py
    ├─ plot_bids_pipeline.py
    ├─ plot_artifact_removal.py
    ├─ plot_ica_and_iclabel.py
    ├─ plot_channel_interpolation.py
    └─ plot_mne_integration.py
    ↓
sphinx-gallery Processing
    ├─ Parse script metadata
    ├─ Extract docstring
    ├─ Execute script
    ├─ Capture output
    ├─ Generate plots
    └─ Create thumbnails
    ↓
Gallery Generation
    ├─ Create gallery index
    ├─ Generate HTML pages
    ├─ Create downloadable .py files
    ├─ Create Jupyter notebooks
    └─ Generate thumbnails
    ↓
Output Structure
    ├─ auto_examples/
    │   ├─ index.html
    │   ├─ plot_basic_preprocessing.html
    │   ├─ plot_bids_pipeline.html
    │   ├─ plot_artifact_removal.html
    │   ├─ plot_ica_and_iclabel.html
    │   ├─ plot_channel_interpolation.html
    │   └─ plot_mne_integration.html
    └─ Downloadable files
        ├─ .py scripts
        └─ .ipynb notebooks
```

## 6. Theme Configuration Hierarchy

```
pydata-sphinx-theme
    ├─ Base Theme
    │   ├─ HTML structure
    │   ├─ Default CSS
    │   └─ JavaScript
    │
    ├─ Configuration (conf.py)
    │   ├─ Logo and branding
    │   ├─ Color scheme
    │   ├─ Navigation structure
    │   ├─ Sidebar configuration
    │   └─ Footer content
    │
    ├─ Custom CSS (_static/custom.css)
    │   ├─ Color overrides
    │   ├─ Font customization
    │   ├─ Layout adjustments
    │   └─ Dark mode tweaks
    │
    ├─ Custom Templates (_templates/)
    │   ├─ Header customization
    │   ├─ Footer customization
    │   └─ Sidebar customization
    │
    └─ Output
        ├─ Responsive HTML
        ├─ Dark mode support
        ├─ Mobile-friendly layout
        └─ Integrated search
```

## 7. CI/CD and Deployment Pipeline

```
GitHub Repository
    ↓
    ├─ Push to main/develop
    │   ↓
    │   GitHub Actions Workflow (.github/workflows/docs.yml)
    │   ├─ Checkout code
    │   ├─ Setup Python environment
    │   ├─ Install dependencies (pip install -e ".[docs]")
    │   ├─ Build documentation (make html)
    │   ├─ Run link checker (make linkcheck)
    │   ├─ Run spell checker (make spelling)
    │   └─ Upload artifacts
    │   ↓
    │   ├─ Success → Deploy to ReadTheDocs
    │   └─ Failure → Notify maintainers
    │
    └─ Pull Request
        ↓
        GitHub Actions Workflow
        ├─ Build documentation
        ├─ Run quality checks
        ├─ Comment on PR with status
        └─ Block merge if checks fail
    ↓
ReadTheDocs
    ├─ Webhook trigger
    ├─ Clone repository
    ├─ Install dependencies
    ├─ Build documentation
    ├─ Run quality checks
    ├─ Generate versioned docs
    └─ Deploy to CDN
    ↓
Output
    ├─ https://eegprep.readthedocs.io/
    ├─ Version switcher
    ├─ Search functionality
    └─ PDF/EPUB downloads
```

## 8. Documentation Content Organization

```
Documentation Root
    ├─ Home Page (index.rst)
    │   ├─ Project overview
    │   ├─ Quick links
    │   ├─ Feature highlights
    │   └─ Installation quick start
    │
    ├─ User Guide (user_guide/)
    │   ├─ Installation
    │   ├─ Quick start
    │   ├─ Preprocessing pipeline
    │   ├─ BIDS workflow
    │   ├─ Configuration
    │   └─ Advanced topics
    │
    ├─ API Reference (api/)
    │   ├─ Core functions
    │   ├─ Preprocessing
    │   ├─ ICA and components
    │   ├─ Signal processing
    │   ├─ I/O functions
    │   └─ Utilities
    │
    ├─ Examples (examples/)
    │   ├─ Basic preprocessing
    │   ├─ BIDS pipeline
    │   ├─ Artifact removal
    │   ├─ ICA and ICLabel
    │   ├─ Channel interpolation
    │   └─ MNE integration
    │
    ├─ Contributing (contributing.rst)
    │   ├─ How to contribute
    │   ├─ Code style
    │   ├─ Testing
    │   └─ PR process
    │
    ├─ Development (development.rst)
    │   ├─ Setup environment
    │   ├─ Running tests
    │   ├─ Building docs
    │   └─ Release process
    │
    ├─ FAQ (faq.rst)
    │   ├─ Common questions
    │   ├─ Troubleshooting
    │   └─ Performance tips
    │
    ├─ References (references.rst)
    │   ├─ Publications
    │   ├─ Related tools
    │   └─ External resources
    │
    ├─ Changelog (changelog.rst)
    │   └─ Version history
    │
    └─ Glossary (glossary.rst)
        └─ EEG terminology
```

## 9. Quality Assurance Workflow

```
Documentation Build
    ↓
    ├─ Sphinx Build
    │   ├─ Parse all source files
    │   ├─ Extract docstrings
    │   ├─ Generate API docs
    │   ├─ Build gallery examples
    │   └─ Check for warnings
    │
    ├─ Link Checking (sphinx-linkcheck)
    │   ├─ Validate internal links
    │   ├─ Check external links
    │   ├─ Report broken links
    │   └─ Generate link report
    │
    ├─ Spell Checking (sphinxcontrib-spelling)
    │   ├─ Check all text
    │   ├─ Use custom dictionary
    │   ├─ Report misspellings
    │   └─ Generate spell report
    │
    ├─ Docstring Validation
    │   ├─ Check docstring format
    │   ├─ Verify parameter docs
    │   ├─ Check return docs
    │   └─ Validate examples
    │
    └─ Output
        ├─ Build log
        ├─ Link report
        ├─ Spell report
        ├─ Docstring report
        └─ Overall status
```

## 10. Deployment Targets

```
Documentation Build Output
    ├─ HTML files
    ├─ Static assets
    ├─ Search index
    └─ Metadata
    ↓
    ├─ ReadTheDocs (Primary)
    │   ├─ Automatic builds on push
    │   ├─ Version management
    │   ├─ CDN distribution
    │   ├─ Search functionality
    │   └─ Analytics
    │
    ├─ GitHub Pages (Optional)
    │   ├─ GitHub Actions deployment
    │   ├─ Custom domain support
    │   ├─ HTTPS enabled
    │   └─ Version switcher
    │
    └─ Local Development
        ├─ Local server (make serve)
        ├─ Live reload
        ├─ Debugging
        └─ Testing
```

## 11. Dependency Graph

```
eegprep (package)
    ├─ Core dependencies
    │   ├─ numpy
    │   ├─ scipy
    │   ├─ mne
    │   ├─ matplotlib
    │   └─ ...
    │
    └─ Documentation dependencies [docs]
        ├─ sphinx (>=7.0)
        ├─ pydata-sphinx-theme (>=0.14.0)
        ├─ sphinx-gallery (>=0.14.0)
        ├─ sphinx-autodoc-typehints (>=1.25.0)
        ├─ numpydoc (>=1.6.0)
        ├─ sphinx-design (>=0.5.0)
        ├─ myst-parser (>=1.0.0)
        ├─ sphinx-copybutton (>=0.5.0)
        ├─ sphinx-togglebutton (>=0.3.0)
        └─ sphinxcontrib-spelling (>=7.1.0)
```

## 12. Documentation Versioning Strategy

```
GitHub Releases
    ├─ v0.2.23 (current stable)
    ├─ v0.2.22
    ├─ v0.2.21
    └─ ...
    ↓
ReadTheDocs Versions
    ├─ latest (development branch)
    ├─ stable (latest release)
    ├─ v0.2.23 (tagged release)
    ├─ v0.2.22 (tagged release)
    └─ ...
    ↓
Documentation URLs
    ├─ https://eegprep.readthedocs.io/en/latest/
    ├─ https://eegprep.readthedocs.io/en/stable/
    ├─ https://eegprep.readthedocs.io/en/v0.2.23/
    └─ https://eegprep.readthedocs.io/en/v0.2.22/
    ↓
Version Switcher
    └─ Dropdown menu to switch between versions
```

---

These diagrams provide a comprehensive visual representation of the documentation infrastructure architecture, showing how all components interact and flow together to create a professional documentation system.
