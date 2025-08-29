# How to Setup PyPI Publishing with GitHub Actions

A comprehensive guide for setting up automated PyPI package publishing using GitHub Actions, based on 2025 best practices and real-world experience.

## 🎯 Key Considerations & Pitfalls

### 1. **Choose Your Authentication Method Carefully**

**Option A: API Token Authentication (Simpler Setup)**
```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}
    attestations: false  # Required with API tokens
```

**Option B: Trusted Publishing (More Secure, 2025 Recommended)**
```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  # No password needed - uses OIDC
```

⚠️ **CRITICAL**: Cannot mix both methods. If using API tokens, must set `attestations: false`.

### 2. **Workflow Structure: Separate Build & Publish Jobs**

```yaml
jobs:
  build:
    name: Build distribution
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: python -m build
      - name: Store artifacts
        uses: actions/upload-artifact@v4
        with:
          name: python-package-distributions
          path: dist/

  publish-to-pypi:
    name: Publish to PyPI
    needs: [build]
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/YOUR-PACKAGE-NAME
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: python-package-distributions
          path: dist/
      - name: Publish
        uses: pypa/gh-action-pypi-publish@release/v1
```

### 3. **Trigger Configuration Gotchas**

```yaml
on:
  push:
    tags:
      - 'v*'  # Triggers on version tags like v1.0.0
  release:
    types: [published]  # Alternative: trigger on GitHub releases
```

⚠️ **PITFALL**: Re-pushing the same tag won't trigger the workflow. You need a new tag version.

### 4. **Environment Setup & Secrets**

```bash
# Set up PyPI API token as GitHub secret
gh secret set PYPI_API_TOKEN --body "pypi-AgEIc..."

# Create GitHub environment for security
gh api repos/:owner/:repo/environments/pypi -X PUT
```

### 5. **Common Workflow Syntax Errors**

❌ **Wrong**: Cannot reference secrets in `if` conditions
```yaml
if: secrets.TEST_PYPI_API_TOKEN != ''  # This fails!
```

✅ **Correct**: Use environment variables or job-level conditions
```yaml
if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
```

### 6. **GitHub Release Integration**

```yaml
github-release:
  name: Create GitHub Release
  needs: [publish-to-pypi]
  runs-on: ubuntu-latest
  permissions:
    contents: write
    id-token: write  # For Sigstore signing
  steps:
    - name: Download artifacts
      uses: actions/download-artifact@v4
    - name: Sign with Sigstore
      uses: sigstore/gh-action-sigstore-python@v3.0.0
    - name: Create release
      run: |
        gh release create '${{ github.ref_name }}' \
          --title 'Release ${{ github.ref_name }}' \
          --notes 'Automated release' \
          --repo '${{ github.repository }}'
```

## 🛠️ Step-by-Step Setup

### 1. Project Prerequisites

Ensure your `pyproject.toml` has correct metadata:
```toml
[project]
name = "your-package-name"
version = "1.0.0"
description = "Your package description"
authors = [{name = "Your Name", email = "you@example.com"}]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 2. Create Workflow File

Create `.github/workflows/publish-pypi.yml`:
```yaml
name: Publish to PyPI

on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Build
        run: |
          python -m pip install build
          python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: python-package-distributions
          path: dist/

  publish-to-pypi:
    needs: [build]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    environment:
      name: pypi
      url: https://pypi.org/p/your-package-name
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: python-package-distributions
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
          attestations: false
```

### 3. Security Setup

```bash
# Get PyPI API token from https://pypi.org/manage/account/token/
# Set as GitHub secret
gh secret set PYPI_API_TOKEN --body "your-token-here"

# Create secure environment
gh api repos/:owner/:repo/environments/pypi -X PUT
```

### 4. Release Process

```bash
# Update version in pyproject.toml
# Commit changes
git add pyproject.toml
git commit -m "bump version to 1.0.1"
git push origin main

# Create and push tag
git tag v1.0.1
git push origin v1.0.1

# Workflow triggers automatically
```

## 🔍 Testing & Verification

### 1. Check Workflow Execution
```bash
gh run list --limit 5
gh run watch <run-id>
```

### 2. Verify Package Publication
```bash
# Wait 30 seconds for PyPI indexing
pip install your-package-name==1.0.1 -i https://pypi.org/simple/
```

### 3. Monitor for Common Issues
- Build failures: Check Python version compatibility
- Authentication errors: Verify API token hasn't expired
- Upload errors: Check package name conflicts

## 🛡️ Security Best Practices

1. **Use GitHub Environments** with approval requirements for production
2. **Separate build/publish jobs** to prevent partial releases  
3. **Enable Sigstore signing** for supply chain security
4. **Use minimal permissions** in workflow jobs
5. **Consider Trusted Publishing** for enhanced security

## 📚 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Silent workflow failures | Check `standalone_mode` in Click CLI tools |
| Tag not triggering workflow | Use new version tag, don't reuse existing tags |
| Mixed authentication errors | Don't use both API tokens and Trusted Publishing |
| Permission errors | Ensure proper `permissions` block in workflow |
| Package not found after upload | Wait for PyPI indexing (~30 seconds) |

## 📖 References

- [Official PyPA GitHub Action](https://github.com/pypa/gh-action-pypi-publish)
- [Python Packaging Guide](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Trusted Publishing Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions Security](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)

---
*Created based on real-world experience setting up PyPI publishing for production packages and official PyPA recommendations.*