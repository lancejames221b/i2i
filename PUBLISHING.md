# Publishing to PyPI

This document explains how to publish the `i2i-mcip` package to PyPI.

## Prerequisites

Before publishing, ensure:
1. You have maintainer access to the PyPI project
2. The package version has been updated in `pyproject.toml`
3. All tests pass
4. The changelog has been updated

## Publishing Methods

### Method 1: Automated Publishing via GitHub Releases (Recommended)

The easiest way to publish is to create a GitHub release. This will automatically trigger the publishing workflow.

1. **Update the version** in `pyproject.toml`:
   ```toml
   [project]
   name = "i2i-mcip"
   version = "0.2.0"  # Update this
   ```

2. **Commit and push** your changes:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.2.0"
   git push
   ```

3. **Create a new release** on GitHub:
   - Go to https://github.com/lancejames221b/i2i/releases/new
   - Create a new tag (e.g., `v0.2.0`)
   - Fill in the release title and description
   - Click "Publish release"

4. **Monitor the workflow**:
   - Go to the Actions tab
   - Watch the "Publish to PyPI" workflow
   - Once complete, verify the package on https://pypi.org/project/i2i-mcip/

### Method 2: Manual Publishing

If you need to publish manually:

1. **Install build tools**:
   ```bash
   pip install build twine
   ```

2. **Build the package**:
   ```bash
   python -m build
   ```

3. **Check the distribution**:
   ```bash
   twine check dist/*
   ```

4. **Upload to PyPI**:
   ```bash
   # For TestPyPI (testing):
   twine upload --repository testpypi dist/*
   
   # For production PyPI:
   twine upload dist/*
   ```

## Setting Up Trusted Publishing (First Time Only)

For automated publishing to work, you need to configure trusted publishing on PyPI:

1. **Go to PyPI**:
   - Visit https://pypi.org/manage/account/publishing/
   - Log in with your PyPI account

2. **Add a new publisher**:
   - Project name: `i2i-mcip`
   - Owner: `lancejames221b`
   - Repository name: `i2i`
   - Workflow name: `publish-to-pypi.yml`
   - Environment name: (leave blank)

3. **Save** the publisher configuration

This allows GitHub Actions to publish without needing API tokens.

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backwards compatible
- **PATCH** (0.0.1): Bug fixes

## Testing the Package

Before publishing, test the package locally:

```bash
# Build the package
python -m build

# Install locally
pip install dist/i2i_mcip-*.whl

# Test imports
python -c "from i2i import AICP; print('Success!')"

# Uninstall
pip uninstall i2i-mcip
```

## Troubleshooting

### Package Already Exists
- You cannot re-upload the same version
- Bump the version number and rebuild

### Authentication Errors
- For manual publishing, ensure your PyPI credentials are correct
- For automated publishing, verify trusted publishing is configured

### Build Errors
- Ensure all dependencies are listed in `pyproject.toml`
- Check that `MANIFEST.in` includes all necessary files

## Post-Publishing Checklist

After publishing:
- [ ] Verify the package appears on https://pypi.org/project/i2i-mcip/
- [ ] Test installation: `pip install i2i-mcip`
- [ ] Update GitHub release notes if needed
- [ ] Announce the release (Twitter, blog, etc.)
