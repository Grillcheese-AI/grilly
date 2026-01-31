# PyPI Submission Guide for Grilly

This guide explains how to publish Grilly to PyPI (Python Package Index).

## Quick Start with Makefile

```bash
# 1. Compile shaders
make compile-shaders

# 2. Run tests
make test

# 3. Build package
make build

# 4. Publish to Test PyPI
make publish-test

# 5. Publish to production PyPI
make publish
```

For detailed step-by-step instructions, continue reading below.

## Prerequisites

### 1. Install Build Tools

```bash
# Using uv (recommended)
uv pip install --upgrade build twine

# Or using pip
pip install --upgrade build twine
```

### 2. Create PyPI Account

- **Production PyPI**: https://pypi.org/account/register/
- **Test PyPI** (recommended first): https://test.pypi.org/account/register/

### 3. Set Up API Tokens

1. Go to PyPI Account Settings → API tokens
2. Create a new API token with scope for "Entire account" or "grilly" project
3. Save the token securely (you'll only see it once)

### 4. Configure `.pypirc`

Create/edit `~/.pypirc` (Windows: `C:\Users\<username>\.pypirc`):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

## Pre-Publication Checklist

### ✅ Required Files

- [x] `pyproject.toml` - Package metadata and dependencies
- [x] `README.md` - Package description (shown on PyPI)
- [x] `LICENSE` - MIT License
- [x] `CHANGELOG.md` - Version history
- [x] `MANIFEST.in` - Include/exclude rules for package files

### ✅ Version Management

Update version in `pyproject.toml`:

```toml
[project]
name = "grilly"
version = "0.1.0"  # Update this for each release
```

Version format: `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### ✅ Compile Shaders

Ensure all shaders are compiled to `.spv`:

```bash
# Using Makefile (recommended)
make compile-shaders
make verify-shaders

# Or manually:
# Windows: .\scripts\compile_all_shaders.ps1
# Linux/Mac: ./compile_shaders.sh
```

### ✅ Run Tests

```bash
# All tests
make test

# CPU-only tests (for CI)
make test-cpu

# With coverage
make test-coverage

# Ensure all tests pass before publishing
```

### ✅ Check Package Metadata

```bash
# Validate pyproject.toml
uv run python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"

# Check for common issues
uv run python -m build --check
```

## Building the Package

### 1. Build Distribution

```bash
# Using Makefile (recommended - includes clean)
make build

# Or manually:
make clean
python -m build

# This creates:
# - dist/grilly-0.1.0-py3-none-any.whl (wheel)
# - dist/grilly-0.1.0.tar.gz (source)
```

### 3. Verify Package Contents

```bash
# Check wheel contents
unzip -l dist/grilly-0.1.0-py3-none-any.whl

# Ensure shaders are included:
# - grilly/shaders/*.glsl
# - grilly/shaders/spv/*.spv
```

## Publishing to PyPI

### Step 1: Test on Test PyPI (Recommended)

```bash
# Using Makefile
make publish-test

# Or manually:
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ grilly

# Verify import works
python -c "import grilly; print(grilly.__version__)"
```

### Step 2: Publish to Production PyPI

```bash
# Using Makefile (includes confirmation prompt)
make publish

# Or manually:
twine upload dist/*

# Installation will now work with:
# pip install grilly
```

## Post-Publication

### 1. Tag Release on Git

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

### 2. Create GitHub Release

1. Go to repository → Releases → Draft a new release
2. Tag: `v0.1.0`
3. Title: `Grilly v0.1.0`
4. Description: Copy from `CHANGELOG.md`
5. Attach: `dist/grilly-0.1.0.tar.gz` and `.whl` files

### 3. Update Documentation

- Update PyPI page description (edit on PyPI if needed)
- Update installation instructions in `README.md`
- Announce on relevant channels (Discord, Reddit, etc.)

### 4. Verify Installation

```bash
# Create fresh environment
python -m venv test-env
source test-env/bin/activate  # Windows: test-env\Scripts\activate

# Install from PyPI
pip install grilly

# Test basic import
python -c "import grilly; print(grilly.__version__)"
```

## Updating grillcheese_model Dependency

After publishing to PyPI, update `grillcheese_model/pyproject.toml`:

```toml
[project]
dependencies = [
    "grilly>=0.1.0",  # Add this instead of local path
    "numpy>=1.24.0",
    "fastapi>=0.109.0",
    # ... other deps
]
```

Then reinstall:

```bash
cd grillcheese_model
uv sync
```

## Troubleshooting

### "File already exists" Error

If you get this error, you're trying to upload the same version twice. PyPI doesn't allow overwriting releases.

**Solution**: Increment version number in `pyproject.toml` and rebuild.

### Shaders Not Included

If shaders aren't in the wheel:

1. Check `MANIFEST.in` includes shader patterns
2. Verify `pyproject.toml` has `package-data` configuration
3. Ensure shaders are in `grilly/shaders/` directory (not project root)

### Import Errors After Installation

Common causes:
- Missing dependencies (check `pyproject.toml`)
- Vulkan not installed on target system
- Shader files not included in package

**Debug**:
```bash
python -c "import grilly; print(grilly.__file__)"  # Check install location
ls -R $(python -c "import grilly; print(grilly.__file__)" | sed 's/__init__.py//')/shaders/
```

## CI/CD Integration (Future)

### GitHub Actions Example

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

## Version Roadmap

- **0.1.0** (Current): Initial release
- **0.2.0** (Next): PLIF neurons, ANN→SNN conversion
- **0.3.0**: Multi-GPU support
- **1.0.0**: Stable API, production-ready

## Useful Commands

```bash
# Check package is on PyPI
curl https://pypi.org/pypi/grilly/json | python -m json.tool

# Download statistics (after some time)
# Use pypistats: pip install pypistats
pypistats recent grilly

# Check reverse dependencies
pip-ecosystem-dependencies grilly
```

## Support

- **Issues**: https://github.com/grillcheese-ai/grilly/issues
- **Discussions**: https://github.com/grillcheese-ai/grilly/discussions
- **PyPI**: https://pypi.org/project/grilly/

---

**Ready to publish?** Follow the steps above and Grilly will be available to everyone via `pip install grilly`!
