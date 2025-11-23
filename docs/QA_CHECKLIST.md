# Quality Assurance Checklist

This document provides comprehensive checklists for documentation quality assurance, covering pre-release verification, content review, and deployment validation.

## Table of Contents

- [Pre-Release Checklist](#pre-release-checklist)
- [Documentation Review Checklist](#documentation-review-checklist)
- [Link Validation Checklist](#link-validation-checklist)
- [Spell Checking Checklist](#spell-checking-checklist)
- [Build Verification Checklist](#build-verification-checklist)
- [Deployment Verification Checklist](#deployment-verification-checklist)
- [Post-Release Checklist](#post-release-checklist)

## Pre-Release Checklist

### Code Quality

- [ ] All source code follows PEP 8 style guide
- [ ] No unused imports or variables
- [ ] All functions have docstrings
- [ ] Docstrings follow Google style format
- [ ] Type hints are present for all parameters
- [ ] No hardcoded values or magic numbers
- [ ] Error handling is appropriate
- [ ] Logging is used appropriately

### Testing

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code coverage is above 80%
- [ ] No failing tests in CI/CD pipeline
- [ ] Edge cases are tested
- [ ] Error conditions are tested
- [ ] Performance tests pass
- [ ] No memory leaks detected

### Documentation

- [ ] All public APIs are documented
- [ ] All examples are runnable
- [ ] All code examples are tested
- [ ] Changelog is updated
- [ ] Version numbers are updated
- [ ] README is current
- [ ] Installation instructions are accurate
- [ ] Contributing guidelines are clear

### Dependencies

- [ ] All dependencies are listed in requirements files
- [ ] Dependency versions are pinned appropriately
- [ ] No deprecated dependencies
- [ ] No security vulnerabilities in dependencies
- [ ] Dependency compatibility is verified
- [ ] Optional dependencies are clearly marked

## Documentation Review Checklist

### Content Accuracy

- [ ] All information is accurate and current
- [ ] Examples produce expected output
- [ ] Code examples follow best practices
- [ ] API documentation matches implementation
- [ ] Parameter descriptions are accurate
- [ ] Return value descriptions are accurate
- [ ] Exception documentation is complete
- [ ] No outdated information

### Completeness

- [ ] All public functions are documented
- [ ] All classes are documented
- [ ] All modules are documented
- [ ] All parameters are documented
- [ ] All return values are documented
- [ ] All exceptions are documented
- [ ] All examples are provided
- [ ] Related functions are linked

### Clarity and Readability

- [ ] Language is clear and concise
- [ ] Sentences are short and simple
- [ ] Paragraphs are well-organized
- [ ] Headings are descriptive
- [ ] Technical terms are explained
- [ ] Abbreviations are defined
- [ ] Active voice is used
- [ ] No jargon without explanation

### Consistency

- [ ] Terminology is consistent
- [ ] Formatting is consistent
- [ ] Style matches style guide
- [ ] Tone is consistent
- [ ] Examples follow same pattern
- [ ] Code style is consistent
- [ ] Heading hierarchy is consistent
- [ ] Abbreviations are consistent

### Organization

- [ ] Content is logically organized
- [ ] Sections flow naturally
- [ ] Related content is grouped
- [ ] Navigation is clear
- [ ] Table of contents is accurate
- [ ] Cross-references are appropriate
- [ ] No duplicate content
- [ ] Outline is balanced

### Accessibility

- [ ] Images have alt text
- [ ] Color is not the only indicator
- [ ] Contrast is sufficient
- [ ] Font size is readable
- [ ] Links are descriptive
- [ ] Code examples are accessible
- [ ] Tables are properly formatted
- [ ] Lists are properly formatted

## Link Validation Checklist

### Internal Links

- [ ] All internal links are valid
- [ ] Links use relative paths
- [ ] Links point to correct sections
- [ ] Anchor links work correctly
- [ ] No circular references
- [ ] Links are not broken by refactoring
- [ ] Cross-references are bidirectional
- [ ] Links are tested in all browsers

### External Links

- [ ] All external links are valid
- [ ] Links point to current URLs
- [ ] Links are to authoritative sources
- [ ] Links are not to deprecated pages
- [ ] Links use HTTPS when available
- [ ] Links are tested regularly
- [ ] Broken links are reported
- [ ] Link text is descriptive

### Link Checking Tools

```bash
# Check links locally
sphinx-linkcheck -b linkcheck docs/source docs/_build/linkcheck

# Check external links
linkchecker --check-extern docs/_build/html/

# Automated link checking in CI/CD
# See .github/workflows/docs-qa.yml
```

## Spell Checking Checklist

### Spelling

- [ ] No spelling errors in documentation
- [ ] No spelling errors in code comments
- [ ] No spelling errors in docstrings
- [ ] No spelling errors in examples
- [ ] Consistent spelling of technical terms
- [ ] Consistent capitalization
- [ ] No typos in URLs
- [ ] No typos in code

### Grammar

- [ ] No grammatical errors
- [ ] Sentences are complete
- [ ] Subject-verb agreement is correct
- [ ] Tense is consistent
- [ ] Punctuation is correct
- [ ] No run-on sentences
- [ ] No sentence fragments
- [ ] Proper use of articles (a/an/the)

### Spell Checking Tools

```bash
# Install spell checker
pip install pyspelling

# Check spelling
pyspelling -c .spellcheckrc

# Interactive spell checking
aspell check docs/source/index.rst

# Automated spell checking in CI/CD
# See .github/workflows/docs-qa.yml
```

### Custom Dictionary

Create `.spellcheckrc` for project-specific terms:

```
matrix:
  - name: markdown
    sources:
      - 'docs/**/*.md'
    aspell:
      lang: en
    dictionary:
      wordlists:
        - .spelling
      output: .spelling.out
```

Create `.spelling` file with project-specific terms:

```
eegprep
EEG
ICA
ASR
BIDS
MNE
```

## Build Verification Checklist

### Local Build

- [ ] Documentation builds without errors
- [ ] Documentation builds without warnings
- [ ] All examples execute successfully
- [ ] All images are included
- [ ] All CSS is applied correctly
- [ ] All JavaScript is loaded
- [ ] HTML is valid
- [ ] No broken references

### Build Commands

```bash
# Clean build
cd docs
make clean

# Build HTML
make html

# Build PDF (if configured)
make pdf

# Check for warnings
make html 2>&1 | grep -i warning

# Validate HTML
html5validator docs/_build/html/
```

### Build Performance

- [ ] Build completes in reasonable time
- [ ] No memory issues during build
- [ ] No disk space issues
- [ ] Build is reproducible
- [ ] Incremental builds work
- [ ] Clean builds work

### ReadTheDocs Build

- [ ] Build succeeds on ReadTheDocs
- [ ] No warnings in ReadTheDocs build
- [ ] All versions build successfully
- [ ] PDF builds successfully (if enabled)
- [ ] Search index is built
- [ ] Version switcher works

## Deployment Verification Checklist

### Pre-Deployment

- [ ] All tests pass
- [ ] All documentation builds successfully
- [ ] No broken links
- [ ] No spelling errors
- [ ] Version numbers are updated
- [ ] Changelog is updated
- [ ] Release notes are prepared
- [ ] Deployment plan is reviewed

### Deployment

- [ ] Code is merged to main branch
- [ ] Release tag is created
- [ ] ReadTheDocs build is triggered
- [ ] Build completes successfully
- [ ] Documentation is published
- [ ] Version switcher is updated
- [ ] Stable version is set correctly
- [ ] Custom domain resolves

### Post-Deployment

- [ ] Documentation is accessible
- [ ] All pages load correctly
- [ ] Search functionality works
- [ ] Version switcher works
- [ ] Mobile rendering is correct
- [ ] Links are not broken
- [ ] Examples are executable
- [ ] Analytics are tracking

### Deployment Checklist

```bash
# Verify documentation is live
curl -I https://eegprep.readthedocs.io/

# Check version switcher
curl https://eegprep.readthedocs.io/ | grep -i version

# Verify custom domain
curl -I https://docs.example.com/

# Check SSL certificate
openssl s_client -connect eegprep.readthedocs.io:443

# Test search functionality
# Manually test in browser
```

## Post-Release Checklist

### Monitoring

- [ ] Monitor for build failures
- [ ] Monitor for broken links
- [ ] Monitor for user issues
- [ ] Monitor analytics
- [ ] Monitor error logs
- [ ] Monitor performance metrics
- [ ] Monitor user feedback

### Maintenance

- [ ] Fix any reported issues
- [ ] Update documentation as needed
- [ ] Respond to user questions
- [ ] Monitor for security issues
- [ ] Update dependencies
- [ ] Archive old versions
- [ ] Update version switcher

### Communication

- [ ] Announce release
- [ ] Share release notes
- [ ] Update social media
- [ ] Notify stakeholders
- [ ] Update project status
- [ ] Respond to feedback
- [ ] Document lessons learned

## Automated QA

### CI/CD Pipeline

The `.github/workflows/docs-qa.yml` workflow automates many QA checks:

- Docstring validation
- Link checking
- Spell checking
- Build verification
- Scheduled weekly runs

### Running QA Locally

```bash
# Run all QA checks
make qa

# Run specific checks
make lint
make test
make docs
make linkcheck
make spellcheck
```

### QA Configuration Files

- `.spellcheckrc`: Spell checker configuration
- `.spelling`: Custom dictionary
- `pyproject.toml`: Linting configuration
- `.github/workflows/docs-qa.yml`: CI/CD workflow

## QA Metrics

### Documentation Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Code coverage | > 80% | - |
| Documentation coverage | 100% | - |
| Broken links | 0 | - |
| Spelling errors | 0 | - |
| Build warnings | 0 | - |
| Example success rate | 100% | - |

### Build Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Build time | < 5 min | - |
| Build success rate | 100% | - |
| Test pass rate | 100% | - |
| Deployment success rate | 100% | - |

## Troubleshooting QA Issues

### Build Failures

1. Check build logs
2. Verify dependencies are installed
3. Check for Python version compatibility
4. Verify configuration files
5. Try clean build

### Link Failures

1. Verify link syntax
2. Check file paths
3. Verify external URLs
4. Check for typos
5. Test in browser

### Spell Check Failures

1. Check spelling
2. Add to custom dictionary if correct
3. Verify dictionary is loaded
4. Check for false positives
5. Update spell checker

### Example Failures

1. Run example locally
2. Check for missing imports
3. Verify data files exist
4. Check for version compatibility
5. Update example if needed

## References

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [ReadTheDocs Documentation](https://docs.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Testing Best Practices](https://docs.pytest.org/)
