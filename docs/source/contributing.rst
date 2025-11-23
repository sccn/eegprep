.. _contributing:

=======================
Contributing to EEGPrep
=======================

We welcome contributions from the community! This guide will help you get started with contributing to EEGPrep.

Getting Started
===============

Fork and Clone
--------------

1. Fork the repository on GitHub by clicking the "Fork" button
2. Clone your fork locally:

.. code-block:: bash

    git clone https://github.com/YOUR_USERNAME/eegprep.git
    cd eegprep

3. Add the upstream repository:

.. code-block:: bash

    git remote add upstream https://github.com/NeuroTechX/eegprep.git

4. Create a new branch for your feature or bugfix:

.. code-block:: bash

    git checkout -b feature/your-feature-name

Development Environment
=======================

Virtual Environment Setup
--------------------------

Create and activate a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies
--------------------

Install the package in editable mode with development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"
    pip install -r requirements-docs.txt

This installs:

- The eegprep package in editable mode
- Testing dependencies (pytest, pytest-cov)
- Documentation dependencies (sphinx, sphinx-rtd-theme)
- Code quality tools (black, flake8, isort)

Code Style Guidelines
=====================

PEP 8 Compliance
----------------

We follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guidelines. Key points:

- Use 4 spaces for indentation (never tabs)
- Maximum line length: 100 characters
- Use descriptive variable and function names
- Add spaces around operators: ``x = 1``, not ``x=1``

Naming Conventions
------------------

- **Functions and variables**: Use lowercase with underscores (``snake_case``)
- **Classes**: Use CapWords (``PascalCase``)
- **Constants**: Use UPPERCASE with underscores (``CONSTANT_NAME``)
- **Private methods/attributes**: Prefix with underscore (``_private_method``)

Code Formatting
---------------

Use automated tools to maintain consistency:

.. code-block:: bash

    # Format code with black
    black src/eegprep tests

    # Sort imports with isort
    isort src/eegprep tests

    # Check code style with flake8
    flake8 src/eegprep tests

Testing Requirements
====================

Running Tests
-------------

Run the full test suite:

.. code-block:: bash

    pytest

Run tests for a specific module:

.. code-block:: bash

    pytest tests/test_clean_artifacts.py

Run tests with verbose output:

.. code-block:: bash

    pytest -v

Test Coverage
-------------

Check test coverage:

.. code-block:: bash

    pytest --cov=src/eegprep --cov-report=html

View the coverage report:

.. code-block:: bash

    open htmlcov/index.html  # On macOS
    # or
    xdg-open htmlcov/index.html  # On Linux

Writing Tests
-------------

When adding new features, include tests:

.. code-block:: python

    import pytest
    from eegprep import EEGobj

    def test_new_feature():
        """Test description of what this tests."""
        # Setup
        eeg = EEGobj()
        
        # Execute
        result = eeg.new_feature()
        
        # Assert
        assert result is not None
        assert len(result) > 0

Documentation Standards
=======================

Docstring Format
----------------

Use Google-style docstrings:

.. code-block:: python

    def preprocess_eeg(eeg, filter_type='bandpass', freq_range=(1, 50)):
        """Preprocess EEG data with filtering and artifact removal.
        
        This function applies a series of preprocessing steps to clean
        EEG data for further analysis.
        
        Parameters
        ----------
        eeg : EEGobj
            The EEG object to preprocess.
        filter_type : str, optional
            Type of filter to apply. Options: 'bandpass', 'highpass', 'lowpass'.
            Default is 'bandpass'.
        freq_range : tuple, optional
            Frequency range for filtering in Hz. Default is (1, 50).
        
        Returns
        -------
        EEGobj
            The preprocessed EEG object.
        
        Raises
        ------
        ValueError
            If filter_type is not recognized.
        TypeError
            If eeg is not an EEGobj instance.
        
        Examples
        --------
        >>> import eegprep
        >>> eeg = eegprep.EEGobj.load('data.set')
        >>> eeg_clean = eegprep.preprocess_eeg(eeg, freq_range=(1, 50))
        
        Notes
        -----
        This function modifies the EEG object in place and returns it.
        
        See Also
        --------
        clean_artifacts : Remove artifacts from EEG data
        clean_flatlines : Remove flat-line channels
        """
        # Implementation here
        pass

Documentation Examples
----------------------

Include practical examples in docstrings:

.. code-block:: python

    def load_bids_dataset(bids_root, subject_id):
        """Load a BIDS-formatted EEG dataset.
        
        Examples
        --------
        >>> import eegprep
        >>> eeg = eegprep.load_bids_dataset('/data/bids_root', 'sub-001')
        >>> print(eeg.nbchan)  # Number of channels
        64
        """
        pass

Pull Request Process
====================

Before Submitting
-----------------

1. Update your branch with the latest upstream changes:

.. code-block:: bash

    git fetch upstream
    git rebase upstream/main

2. Run tests locally:

.. code-block:: bash

    pytest

3. Check code style:

.. code-block:: bash

    black --check src/eegprep tests
    flake8 src/eegprep tests

4. Build documentation locally:

.. code-block:: bash

    cd docs
    make html

Submitting a Pull Request
--------------------------

1. Push your branch to your fork:

.. code-block:: bash

    git push origin feature/your-feature-name

2. Go to the GitHub repository and click "New Pull Request"

3. Fill in the PR template with:

   - **Title**: Clear, descriptive title
   - **Description**: What changes are made and why
   - **Related Issues**: Link to any related issues (e.g., "Fixes #123")
   - **Testing**: Describe how you tested the changes
   - **Documentation**: Note any documentation updates

4. Ensure all CI checks pass

5. Wait for review and address feedback

PR Review Process
-----------------

- At least one maintainer review is required
- All CI checks must pass
- Code coverage should not decrease
- Documentation must be updated if needed
- Commits should be clean and well-organized

Code of Conduct
===============

Respectful Collaboration
------------------------

We are committed to providing a welcoming and inclusive environment. All contributors must:

- Be respectful and professional in all interactions
- Welcome diverse perspectives and experiences
- Provide constructive feedback
- Report inappropriate behavior to the maintainers

Unacceptable Behavior
---------------------

The following behaviors are not tolerated:

- Harassment, discrimination, or intimidation
- Offensive comments or language
- Unwelcome sexual attention or advances
- Deliberate disruption of discussions
- Publishing private information without consent

Reporting Issues
----------------

If you experience or witness unacceptable behavior, please report it to the maintainers at:

- Open an issue on GitHub (private if needed)
- Contact the project maintainers directly

All reports will be handled confidentially and investigated promptly.

Getting Help
============

- **Documentation**: Check the :doc:`user_guide/index` and :doc:`api/index`
- **Examples**: See :doc:`examples/index` for practical examples
- **Issues**: Search `GitHub Issues <https://github.com/NeuroTechX/eegprep/issues>`_
- **Discussions**: Join our community discussions on GitHub

Thank you for contributing to EEGPrep!
