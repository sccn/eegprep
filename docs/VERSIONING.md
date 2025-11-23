# Documentation Versioning Strategy

This document outlines the versioning strategy for eegprep documentation, including version management, release procedures, and deprecation policies.

## Table of Contents

- [Versioning Overview](#versioning-overview)
- [Version Management](#version-management)
- [Stable vs Latest Versions](#stable-vs-latest-versions)
- [Version Switcher Configuration](#version-switcher-configuration)
- [Release Process](#release-process)
- [Deprecation Policy](#deprecation-policy)
- [Maintenance Schedule](#maintenance-schedule)

## Versioning Overview

### Semantic Versioning

eegprep follows [Semantic Versioning](https://semver.org/) (SemVer):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Documentation Versioning

Documentation versions align with package releases:

- **Development (latest)**: Tracks main branch
- **Stable**: Latest released version
- **Archived**: Previous release versions (maintained for 2 major versions)

### Version Naming Convention

```
v1.0.0          # Release version
v1.0.0-alpha    # Alpha release
v1.0.0-beta     # Beta release
v1.0.0-rc1      # Release candidate
latest          # Development version (main branch)
stable          # Latest stable release
```

## Version Management

### Creating a New Version

1. **Update Version Numbers**
   ```bash
   # Update in src/eegprep/__init__.py
   __version__ = "1.1.0"
   
   # Update in pyproject.toml
   version = "1.1.0"
   ```

2. **Update Changelog**
   - Add entry to `docs/source/changelog.rst`
   - Document all changes, features, and fixes
   - Include migration guide for breaking changes

3. **Create Release Tag**
   ```bash
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0
   ```

4. **ReadTheDocs Automatic Build**
   - ReadTheDocs detects the tag
   - Automatically builds documentation
   - Makes version available in switcher

### Version Branches

```
main                    # Development branch
release/v1.1.x         # Release branch for v1.1.x
release/v1.0.x         # Release branch for v1.0.x (maintenance)
```

### Maintenance Branches

- **Active Development**: main branch
- **Current Release**: release/v1.x.x
- **Previous Release**: release/v1.(x-1).x (bug fixes only)
- **Older Releases**: Archived (no updates)

## Stable vs Latest Versions

### Latest Version

**Purpose**: Development version with latest features

**Characteristics**:
- Tracks main branch
- May contain experimental features
- Documentation may be incomplete
- Not recommended for production use

**URL**: `https://eegprep.readthedocs.io/en/latest/`

**Configuration**:
```python
# In conf.py
version = "latest"
release = "latest"
```

### Stable Version

**Purpose**: Latest released version for production use

**Characteristics**:
- Tracks latest release tag
- Fully tested and documented
- Recommended for production use
- Receives bug fixes and security updates

**URL**: `https://eegprep.readthedocs.io/en/stable/`

**Configuration**:
```python
# In conf.py
version = "1.1.0"
release = "1.1.0"
```

### Setting Stable Version in ReadTheDocs

1. Log in to ReadTheDocs
2. Go to Admin → Versions
3. Find the release version (e.g., v1.1.0)
4. Click the version name
5. Check "Active" and "Public"
6. Go to Admin → Advanced Settings
7. Set "Default Version" to the release version

## Version Switcher Configuration

### Enabling Version Switcher

The version switcher allows users to switch between documentation versions.

**In conf.py**:
```python
html_theme_options = {
    "version_switcher": True,
    "versions": {
        "latest": "https://eegprep.readthedocs.io/en/latest/",
        "stable": "https://eegprep.readthedocs.io/en/stable/",
        "1.1.0": "https://eegprep.readthedocs.io/en/1.1.0/",
        "1.0.0": "https://eegprep.readthedocs.io/en/1.0.0/",
    }
}
```

### Version Switcher Behavior

- Appears in top navigation bar
- Dropdown menu shows available versions
- Current version is highlighted
- Clicking a version navigates to that documentation

### Updating Version Switcher

When releasing a new version:

1. Update `versions` dictionary in conf.py
2. Add new version entry
3. Remove very old versions (keep last 3-4)
4. Rebuild documentation
5. Verify switcher works correctly

## Release Process

### Pre-Release Checklist

- [ ] All tests pass
- [ ] Code review completed
- [ ] Documentation is complete and accurate
- [ ] Changelog is updated
- [ ] Version numbers are updated
- [ ] No breaking changes without migration guide
- [ ] Examples execute without errors
- [ ] Links are not broken

### Release Steps

1. **Create Release Branch**
   ```bash
   git checkout -b release/v1.1.0
   ```

2. **Update Version Numbers**
   - Update `__version__` in `src/eegprep/__init__.py`
   - Update `version` in `pyproject.toml`
   - Update version in `docs/source/conf.py`

3. **Update Changelog**
   ```bash
   # Add to docs/source/changelog.rst
   Version 1.1.0 (2024-01-15)
   ===========================
   
   New Features
   -----------
   - Feature 1
   - Feature 2
   
   Bug Fixes
   ---------
   - Fix 1
   - Fix 2
   
   Breaking Changes
   ----------------
   - Change 1 (migration guide provided)
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Release v1.1.0"
   ```

5. **Create Release Tag**
   ```bash
   git tag -a v1.1.0 -m "Release version 1.1.0"
   ```

6. **Push Changes**
   ```bash
   git push origin release/v1.1.0
   git push origin v1.1.0
   ```

7. **Create Pull Request**
   - Create PR from release branch to main
   - Request review
   - Merge after approval

8. **Update ReadTheDocs**
   - Verify build succeeds
   - Set as stable version
   - Update version switcher

### Post-Release

- [ ] Verify documentation builds successfully
- [ ] Test version switcher
- [ ] Verify stable version is set correctly
- [ ] Announce release in appropriate channels
- [ ] Monitor for issues

## Deprecation Policy

### Deprecation Timeline

**Phase 1: Announcement** (1 release)
- Feature marked as deprecated
- Warning message added to documentation
- Migration guide provided

**Phase 2: Deprecation Warning** (1-2 releases)
- Code emits deprecation warning
- Documentation clearly marks as deprecated
- Users have time to migrate

**Phase 3: Removal** (Next major version)
- Feature is removed
- Breaking change documented
- Migration guide in changelog

### Deprecation Example

```python
# In code
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    # Implementation
```

```rst
.. deprecated:: 1.1.0
   Use :func:`new_function` instead.
```

### Deprecation Documentation

In changelog:
```
Deprecated
----------
- old_function: Use new_function instead (will be removed in v2.0.0)
```

In API documentation:
```rst
.. deprecated:: 1.1.0
   Use :func:`new_function` instead. This function will be removed in version 2.0.0.
```

## Maintenance Schedule

### Version Support Matrix

| Version | Release Date | End of Life | Status |
|---------|-------------|------------|--------|
| 1.1.x   | 2024-01-15  | 2025-01-15 | Active |
| 1.0.x   | 2023-06-01  | 2024-06-01 | Maintenance |
| 0.9.x   | 2023-01-01  | 2023-07-01 | Archived |

### Support Levels

**Active Development**
- Latest version
- New features and improvements
- Bug fixes and security updates
- Duration: Until next major release

**Maintenance**
- Previous version
- Bug fixes and security updates only
- No new features
- Duration: 6-12 months

**Archived**
- Older versions
- No updates
- Documentation available for reference
- Duration: Indefinite

### Bug Fix Policy

- **Critical Security Issues**: Fixed in all active versions
- **Major Bugs**: Fixed in current and previous version
- **Minor Bugs**: Fixed in current version only

### Documentation Maintenance

- **Latest**: Updated with every commit
- **Stable**: Updated with every release
- **Archived**: No updates (frozen at release time)

## Version Compatibility

### Python Version Support

```
eegprep 1.1.x: Python 3.8+
eegprep 1.0.x: Python 3.7+
eegprep 0.9.x: Python 3.6+
```

### Dependency Compatibility

Document minimum versions for key dependencies:

```
numpy >= 1.19.0
scipy >= 1.5.0
mne >= 0.23.0
```

## Troubleshooting

### Version Not Appearing in Switcher

1. Verify version is marked as "Active" in ReadTheDocs
2. Check version is listed in conf.py
3. Rebuild documentation
4. Clear browser cache

### Broken Links Between Versions

1. Use relative paths for internal links
2. Test links in each version
3. Update intersphinx mappings if needed

### Version Switcher Not Working

1. Verify `version_switcher` is enabled in conf.py
2. Check all version URLs are accessible
3. Verify version names match exactly
4. Test in different browsers

## References

- [Semantic Versioning](https://semver.org/)
- [ReadTheDocs Versioning](https://docs.readthedocs.io/en/stable/versions.html)
- [Python Packaging Guide](https://packaging.python.org/)
- [PEP 440 - Version Identification](https://www.python.org/dev/peps/pep-0440/)
