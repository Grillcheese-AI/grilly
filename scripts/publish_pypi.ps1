# Publish grilly to PyPI
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\publish_pypi.ps1
#
# Prerequisites:
#   - Set PYPI_TOKEN env var or have ~/.pypirc configured
#   - uv installed (pip install uv)
#
# This script:
#   1. Cleans old dist/ files
#   2. Builds wheel with C++ extension via scikit-build-core
#   3. Publishes to PyPI

$ErrorActionPreference = "Stop"

Write-Host "=== Grilly PyPI Publisher ===" -ForegroundColor Cyan

# Check for token
$token = $env:PYPI_TOKEN
if (-not $token) {
    $token = $env:UV_PUBLISH_TOKEN
}
if (-not $token) {
    Write-Host "ERROR: Set PYPI_TOKEN or UV_PUBLISH_TOKEN environment variable" -ForegroundColor Red
    Write-Host "  Get your token from: https://pypi.org/manage/account/token/" -ForegroundColor Yellow
    Write-Host "  Then: `$env:PYPI_TOKEN = 'pypi-...'" -ForegroundColor Yellow
    exit 1
}

# Clean
Write-Host "`n[1/3] Cleaning dist/..." -ForegroundColor Yellow
Remove-Item -Path "dist\*" -Force -ErrorAction SilentlyContinue

# Build
Write-Host "[2/3] Building wheel (C++ extension via scikit-build-core)..." -ForegroundColor Yellow
uv build --wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    exit 1
}

$wheel = Get-ChildItem dist\*.whl | Select-Object -First 1
Write-Host "  Built: $($wheel.Name) ($([math]::Round($wheel.Length / 1MB, 1)) MB)" -ForegroundColor Green

# Publish
Write-Host "[3/3] Publishing to PyPI..." -ForegroundColor Yellow
uv publish $wheel.FullName --token $token
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Publish failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Published successfully! ===" -ForegroundColor Green
Write-Host "  pip install grilly==$((Get-Content pyproject.toml | Select-String 'version = "(.+)"' | ForEach-Object { $_.Matches.Groups[1].Value }))" -ForegroundColor Cyan
