# Architecture and Staged Migration Proposal: Plugin Decoupling

## Context
Currently, `eegprep` has heavy optional dependencies (e.g., PyTorch, CUDA libraries) bundled with in-tree implementations of `ICLabel` and `clean_rawdata`. This results in large installation footprints that create significant friction for serverless deployment and automated CI workflows. To resolve this, we propose decoupling these heavy plugins from the core library.

## Architectural Goal
The core `eegprep` package will transition into a lightweight orchestrator, providing entry points and a dynamic extension catalog for plugin discovery, while hosting foundational capabilities (like MNE/SciPy-based operations). Heavy plugins like `ICLabel` and `clean_rawdata` will be moved to separate, dedicated repositories (`eegprep-iclabel` and `eegprep-clean-rawdata`) with their own isolated CI/CD pipelines, test suites, and distribution units.

## Staged Migration Plan

### Phase 1: External Repositories & Distribution
- **Create New Repositories:** Set up `sccn/eegprep-iclabel` and `sccn/eegprep-clean-rawdata`.
- **Migrate Code & Tests:** Transfer the plugin implementations, models, and associated test files (approx. 19,000 lines) from the main `eegprep` repository to these new repositories.
- **Set Up CI/CD:** Establish independent CI workflows in the new repositories to ensure version compatibility, test parity, and model validation.
- **Publish Initial Versions:** Release initial standalone distributions to PyPI to provide concrete packages rather than hypothetical ones.

### Phase 2: Core Compatibility & Catalog Integration
- **Extension Catalog:** Introduce an extension catalog in `eegprep` that handles dynamic discovery of installed plugins via entry points (`eegprep.plugins`).
- **Graceful Degradation:** Update core functions (`pop_iclabel`, `clean_rawdata`) in `eegprep` to display descriptive warnings and installation instructions (e.g., `pip install eegprep-iclabel`) if the corresponding package is missing.
- **Version Compatibility:** Pin known working versions of the external plugins in `eegprep`'s optional dependencies (`[all]`) to maintain a seamless "all-in-one" installation workflow for local researchers.

### Phase 3: Documentation & Communication
- **Migration Guide:** Publish comprehensive install/migration documentation explaining how users can opt-in to heavy dependencies.
- **Update Tutorials:** Adjust `docs/` and `sample_notebooks/` to explicitly demonstrate plugin installation and usage in decoupled workflows.
- **Deprecation Notice:** Issue a deprecation notice in the main `eegprep` repository indicating that in-tree implementations will be removed in a future release.

### Phase 4: Final Deprecation & Removal (Current Blocked PR)
- Once external distributions are stable, documented, and have fully functional CI, we will merge the removal PR (originally PR #244) to delete the in-tree implementations and finalize the decoupling.

## Conclusion
By adopting this staged approach, we guarantee that core commands and user workflows will remain uninterrupted, leveraging the newly distributed packages before any in-tree code is removed.
