# Documentation Deployment Guide

This guide covers deploying the eegprep documentation to various platforms and managing the deployment infrastructure.

## Table of Contents

- [ReadTheDocs Setup](#readthedocs-setup)
- [GitHub Pages Setup](#github-pages-setup)
- [Custom Domain Configuration](#custom-domain-configuration)
- [Versioning Strategy](#versioning-strategy)
- [Monitoring and Alerts](#monitoring-and-alerts)
- [Troubleshooting](#troubleshooting)

## ReadTheDocs Setup

### Prerequisites

- GitHub account with access to the eegprep repository
- ReadTheDocs account (free tier available)
- Admin access to the repository

### Initial Setup

1. **Connect Repository to ReadTheDocs**
   - Visit [ReadTheDocs](https://readthedocs.org)
   - Sign in with your GitHub account
   - Click "Import a Project"
   - Select the eegprep repository
   - Click "Create"

2. **Configure Build Settings**
   - Go to project settings
   - Set Python version to 3.9+
   - Enable PDF builds (optional)
   - Configure build notifications

3. **Set Environment Variables**
   - Navigate to Admin → Environment Variables
   - Add any required environment variables for builds
   - Example: `READTHEDOCS_VERSION` for version-specific builds

### Build Configuration

The `.readthedocs.yml` file in the repository root controls the build process:

```yaml
version: 2
build:
  os: ubuntu-20.04
  tools:
    python: "3.10"
python:
  version: 3.10
  install:
    - requirements: requirements-docs.txt
sphinx:
  configuration: docs/source/conf.py
  fail_on_warning: false
```

### Automatic Builds

ReadTheDocs automatically builds documentation when:
- Code is pushed to the main branch
- Pull requests are created (preview builds)
- Tags are created (version releases)

### Manual Builds

To trigger a manual build:
1. Log in to ReadTheDocs
2. Go to the eegprep project
3. Click "Build Version"
4. Select the version to build
5. Click "Build"

## GitHub Pages Setup

### Alternative Deployment Method

GitHub Pages can be used as an alternative or supplementary deployment platform.

### Configuration Steps

1. **Enable GitHub Pages**
   - Go to repository Settings
   - Navigate to Pages section
   - Select "Deploy from a branch"
   - Choose `gh-pages` branch
   - Select `/root` directory

2. **Automated Deployment Workflow**
   - Use `.github/workflows/docs.yml` to build and deploy
   - Workflow automatically builds docs on push to main
   - Deploys to `gh-pages` branch

3. **Access Documentation**
   - URL: `https://neurotechtx.github.io/eegprep/`
   - Updates automatically after workflow completes

## Custom Domain Configuration

### Setting Up a Custom Domain

1. **Register Domain**
   - Register your domain with a registrar (e.g., GoDaddy, Namecheap)
   - Note the nameservers or DNS settings

2. **Configure DNS Records**

   **For ReadTheDocs:**
   - Add CNAME record pointing to `eegprep.readthedocs.io`
   - Example: `docs.example.com CNAME eegprep.readthedocs.io`

   **For GitHub Pages:**
   - Add CNAME record pointing to `neurotechtx.github.io`
   - Example: `docs.example.com CNAME neurotechtx.github.io`

3. **Update ReadTheDocs Settings**
   - Go to Admin → Domains
   - Add custom domain
   - Enable HTTPS (automatic with Let's Encrypt)

4. **Update GitHub Pages Settings**
   - Go to Settings → Pages
   - Enter custom domain
   - Enable HTTPS

### SSL/TLS Certificate

- ReadTheDocs: Automatic with Let's Encrypt
- GitHub Pages: Automatic with GitHub's certificate
- Renewal: Automatic, no action required

## Versioning Strategy

### Version Management

Documentation versions correspond to package releases:

- **Latest**: Points to main branch (development version)
- **Stable**: Points to latest release tag
- **Archived**: Previous release versions

### Version Switcher

The version switcher in the documentation allows users to switch between versions:

```python
# In conf.py
html_theme_options = {
    "version_switcher": True,
    "versions": {
        "latest": "https://eegprep.readthedocs.io/en/latest/",
        "stable": "https://eegprep.readthedocs.io/en/stable/",
        "1.0": "https://eegprep.readthedocs.io/en/1.0/",
    }
}
```

### Creating a Release

1. **Tag the Release**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

2. **ReadTheDocs Automatic Build**
   - ReadTheDocs detects the tag
   - Automatically builds documentation for the version
   - Makes it available in version switcher

3. **Update Stable Version**
   - Go to ReadTheDocs Admin
   - Set "Stable" version to the latest release tag

## Monitoring and Alerts

### Build Status Monitoring

1. **ReadTheDocs Dashboard**
   - Monitor build status and history
   - View build logs for debugging
   - Check for warnings and errors

2. **GitHub Actions**
   - Monitor workflow runs in Actions tab
   - View logs for each workflow execution
   - Set up notifications for failures

### Email Notifications

**ReadTheDocs:**
- Go to Notifications settings
- Enable email alerts for build failures
- Configure notification recipients

**GitHub:**
- Settings → Notifications
- Enable email for workflow failures
- Configure notification preferences

### Slack Integration

**ReadTheDocs:**
- Go to Integrations
- Add Slack webhook
- Configure notification channels

**GitHub:**
- Use GitHub Actions to send Slack notifications
- Example workflow:
  ```yaml
  - name: Notify Slack
    if: failure()
    uses: slackapi/slack-github-action@v1
    with:
      webhook-url: ${{ secrets.SLACK_WEBHOOK }}
  ```

## Troubleshooting

### Common Issues and Solutions

#### Build Failures

**Issue**: Documentation build fails on ReadTheDocs

**Solutions**:
1. Check build logs in ReadTheDocs dashboard
2. Verify `requirements-docs.txt` is up to date
3. Ensure `.readthedocs.yml` is correctly configured
4. Check for Python version compatibility issues
5. Verify all imports are available in the build environment

#### Missing Dependencies

**Issue**: Build fails with "ModuleNotFoundError"

**Solutions**:
1. Add missing package to `requirements-docs.txt`
2. Rebuild documentation
3. Check for version conflicts between packages

#### Broken Links

**Issue**: Documentation contains broken links

**Solutions**:
1. Run link checker locally: `sphinx-linkcheck`
2. Fix broken links in source files
3. Update external URLs if they've changed
4. Use relative paths for internal links

#### Slow Builds

**Issue**: Documentation builds take too long

**Solutions**:
1. Disable example execution: Set `plot_gallery = False` in conf.py
2. Reduce number of examples
3. Optimize images and assets
4. Check for large file downloads during build

#### Version Switcher Not Working

**Issue**: Version switcher doesn't appear or doesn't work

**Solutions**:
1. Verify `version_switcher` is enabled in conf.py
2. Check that versions are properly configured
3. Ensure all version URLs are accessible
4. Clear browser cache and rebuild

#### Custom Domain Not Working

**Issue**: Custom domain shows 404 or doesn't resolve

**Solutions**:
1. Verify DNS records are correctly configured
2. Wait for DNS propagation (up to 48 hours)
3. Check CNAME record in ReadTheDocs/GitHub settings
4. Verify SSL certificate is valid
5. Test with `nslookup` or `dig` command

### Debug Commands

```bash
# Check DNS resolution
nslookup docs.example.com

# Verify CNAME record
dig docs.example.com CNAME

# Test ReadTheDocs connectivity
curl -I https://eegprep.readthedocs.io

# Build documentation locally
cd docs
make clean html

# Check for broken links
sphinx-linkcheck -b linkcheck . _build/linkcheck
```

### Getting Help

- **ReadTheDocs Support**: https://docs.readthedocs.io/
- **GitHub Pages Help**: https://docs.github.com/en/pages
- **Sphinx Documentation**: https://www.sphinx-doc.org/
- **eegprep Issues**: https://github.com/NeuroTechX/eegprep/issues

## Deployment Checklist

Before deploying documentation:

- [ ] All documentation builds successfully locally
- [ ] No broken links in documentation
- [ ] All examples execute without errors
- [ ] Version numbers are updated
- [ ] Changelog is updated
- [ ] ReadTheDocs build succeeds
- [ ] GitHub Pages build succeeds (if applicable)
- [ ] Custom domain resolves correctly
- [ ] SSL certificate is valid
- [ ] Version switcher works correctly
- [ ] Search functionality works
- [ ] Mobile rendering looks correct
- [ ] All external links are valid
- [ ] Analytics are configured (if applicable)
