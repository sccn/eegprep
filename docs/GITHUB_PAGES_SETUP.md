# GitHub Pages Setup Guide for EEGPrep Documentation

This guide provides step-by-step instructions for configuring and deploying EEGPrep documentation using GitHub Pages.

## Overview

The EEGPrep documentation is automatically built and deployed to GitHub Pages on every push to the `main` branch. The deployment process is handled by GitHub Actions workflows that:

1. Validate example scripts
2. Build the Sphinx documentation
3. Deploy the built HTML to GitHub Pages
4. Make the documentation accessible via a public URL

## Prerequisites

- Repository owner or admin access to the GitHub repository
- GitHub Actions enabled (enabled by default for public repositories)
- Documentation source files in the `docs/` directory
- Sphinx configuration in `docs/source/conf.py`

## Automatic Deployment Setup

### 1. GitHub Actions Workflows

Two workflows handle the documentation deployment:

#### `docs.yml` - Documentation Build and Validation
- **Location**: `.github/workflows/docs.yml`
- **Triggers**: Push to `main`/`develop` branches, pull requests
- **Jobs**:
  - `validate-examples`: Validates Python example syntax
  - `build-docs`: Builds HTML documentation with Sphinx
  - `deploy-pages`: Deploys to GitHub Pages (main branch only)
  - `docs-status`: Reports overall build status

#### `pages.yml` - GitHub Pages Deployment
- **Location**: `.github/workflows/pages.yml`
- **Triggers**: Push to `main` branch, workflow completion
- **Purpose**: Ensures documentation is deployed to GitHub Pages

### 2. Permissions Configuration

The workflows require the following permissions:

```yaml
permissions:
  contents: read
  pages: write
  id-token: write
```

These permissions are automatically configured in the workflow files.

## Enabling GitHub Pages in Repository Settings

### Step 1: Access Repository Settings

1. Navigate to your GitHub repository
2. Click on **Settings** (gear icon)
3. Scroll down to the **Pages** section in the left sidebar

### Step 2: Configure GitHub Pages Source

1. Under **Build and deployment**:
   - **Source**: Select **GitHub Actions**
   - This allows GitHub Actions workflows to deploy to GitHub Pages

2. The documentation will be deployed from the workflow artifacts

### Step 3: Verify Configuration

1. After the first successful deployment, you should see:
   - **Your site is live at**: `https://<username>.github.io/<repository>/`
   - Or for organization repos: `https://<org>.github.io/<repository>/`

2. The deployment status will show in the **Deployments** section

## Custom Domain Setup (Optional)

If you want to use a custom domain for your documentation:

### Step 1: Configure DNS Records

1. Add a CNAME record to your domain registrar pointing to:
   - For user repos: `<username>.github.io`
   - For org repos: `<org>.github.io`

2. Or use A records pointing to GitHub's IP addresses:
   - `185.199.108.153`
   - `185.199.109.153`
   - `185.199.110.153`
   - `185.199.111.153`

### Step 2: Configure in GitHub Pages Settings

1. In the **Pages** section of repository settings
2. Under **Custom domain**, enter your domain name
3. Click **Save**
4. GitHub will automatically create a CNAME file in your repository

### Step 3: Enable HTTPS

1. After DNS is configured, check **Enforce HTTPS**
2. This may take a few minutes to become available
3. GitHub will automatically provision an SSL certificate

## Accessing the Documentation

### Public URL

Once deployed, the documentation is accessible at:

```
https://<username>.github.io/<repository>/
```

For example:
- User repository: `https://baristim.github.io/eegprep/`
- Organization repository: `https://sccn.github.io/eegprep/`

### From Repository

1. Go to your GitHub repository
2. Click on **Deployments** (or **Environments** → **github-pages**)
3. Click on the latest deployment
4. Click **View deployment** to access the live documentation

## Monitoring Deployments

### GitHub Actions Tab

1. Go to your repository
2. Click on **Actions** tab
3. Select the **Documentation Build and Deploy** workflow
4. View recent runs and their status

### Deployment History

1. Go to **Deployments** section in your repository
2. View all past deployments
3. Click on any deployment to see details and access the live site

## Troubleshooting

### Documentation Build Fails

**Problem**: The `build-docs` job fails

**Solutions**:
1. Check the workflow logs in the **Actions** tab
2. Verify all dependencies are listed in `requirements-docs.txt`
3. Ensure `docs/source/conf.py` is properly configured
4. Check for syntax errors in RST files

### Deployment Fails

**Problem**: The `deploy-pages` job fails

**Solutions**:
1. Verify GitHub Pages is enabled in repository settings
2. Check that the **Source** is set to **GitHub Actions**
3. Ensure the workflow has proper permissions (see Permissions Configuration)
4. Check the deployment logs in the **Deployments** section

### Documentation Not Updating

**Problem**: Changes to documentation don't appear on GitHub Pages

**Solutions**:
1. Verify the workflow ran successfully (check **Actions** tab)
2. Clear your browser cache (Ctrl+Shift+Delete or Cmd+Shift+Delete)
3. Wait a few minutes for GitHub Pages to update
4. Check the deployment URL in the **Deployments** section

### Custom Domain Not Working

**Problem**: Custom domain shows 404 or doesn't resolve

**Solutions**:
1. Verify DNS records are correctly configured
2. Wait 24-48 hours for DNS propagation
3. Check that the CNAME file exists in the repository root
4. Ensure HTTPS is enabled in GitHub Pages settings
5. Try accessing via the default GitHub Pages URL first

### Broken Links in Documentation

**Problem**: Links in the documentation are broken

**Solutions**:
1. The workflow runs `make linkcheck` to validate links
2. Check the workflow logs for broken link reports
3. Fix broken links in the RST source files
4. Rebuild and redeploy

## Workflow Configuration Details

### Build Triggers

The documentation builds on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Changes to documentation, source code, or workflow files

### Build Steps

1. **Checkout**: Clone the repository
2. **Setup Python**: Install Python 3.11
3. **Install Dependencies**: Install eegprep with docs extras
4. **Validate Examples**: Check Python example syntax
5. **Build HTML**: Run `make html` in docs directory
6. **Link Check**: Validate all links (non-blocking)
7. **Spell Check**: Check spelling (non-blocking)
8. **Deploy**: Upload to GitHub Pages (main branch only)

### Caching

The workflow uses caching to speed up builds:
- **pip cache**: Caches Python packages
- **Sphinx cache**: Caches Sphinx build artifacts

## Best Practices

1. **Keep Documentation Updated**: Update docs with code changes
2. **Test Locally**: Run `make html` locally before pushing
3. **Use Descriptive Commit Messages**: Makes it easier to track changes
4. **Review PRs**: Check documentation builds in PR checks
5. **Monitor Deployments**: Check the Deployments section regularly
6. **Backup Custom Domain**: Keep DNS records documented

## Advanced Configuration

### Excluding Files from Deployment

To exclude files from the GitHub Pages deployment, add them to `.gitignore`:

```
docs/build/
docs/source/_build/
*.pyc
__pycache__/
```

### Custom Build Configuration

To customize the build process, edit:
- `docs/source/conf.py`: Sphinx configuration
- `docs/Makefile`: Build commands
- `.github/workflows/docs.yml`: GitHub Actions workflow

### Environment Variables

To add environment variables to the build process:

1. Edit `.github/workflows/docs.yml`
2. Add to the build step:
   ```yaml
   env:
     VARIABLE_NAME: value
   ```

## Support and Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [EEGPrep Documentation](https://github.com/sccn/eegprep)

## Maintenance

### Regular Tasks

1. **Weekly**: Monitor deployment status
2. **Monthly**: Review documentation for outdated content
3. **Quarterly**: Update dependencies in `requirements-docs.txt`
4. **Annually**: Review and update this guide

### Updating Workflows

To update the GitHub Actions workflows:

1. Edit `.github/workflows/docs.yml` or `.github/workflows/pages.yml`
2. Test changes locally if possible
3. Commit and push to a feature branch
4. Create a pull request for review
5. Merge after approval

## FAQ

**Q: How often is the documentation updated?**
A: The documentation is updated automatically on every push to the `main` branch.

**Q: Can I deploy from a different branch?**
A: Yes, edit the `on.push.branches` section in `.github/workflows/docs.yml`.

**Q: How do I disable GitHub Pages deployment?**
A: Remove or disable the `deploy-pages` job in `.github/workflows/docs.yml`.

**Q: Can I use a different documentation tool?**
A: Yes, modify the build step in the workflow to use your preferred tool.

**Q: How do I add a custom theme?**
A: Update the Sphinx theme in `docs/source/conf.py`.

**Q: Can I deploy to multiple branches?**
A: Yes, add additional branches to the `on.push.branches` section and modify the deployment condition.

---

**Last Updated**: 2025-11-22
**Maintained By**: EEGPrep Development Team
