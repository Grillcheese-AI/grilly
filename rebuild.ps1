# rebuild.ps1 — one-shot grilly rebuild: compile shaders + build grilly_core + copy .pyd
#
# Usage (from grilly root):
#   ./rebuild.ps1                  # full rebuild: shaders + grilly_core + copy
#   ./rebuild.ps1 -SkipShaders     # skip shader compile (if only C++ changed)
#   ./rebuild.ps1 -SkipBuild       # just compile shaders + copy existing .pyd
#   ./rebuild.ps1 -Clean           # wipe build2 and reconfigure from scratch
#   ./rebuild.ps1 -Verbose         # print each file as it processes
#
# Exits non-zero on shader or build failure so you can chain it in a Makefile
# or git hook if you're feeling fancy.

[CmdletBinding()]
param(
    [switch]$SkipShaders,
    [switch]$SkipBuild,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$grillyRoot = $PSScriptRoot
Set-Location $grillyRoot

function Write-Step {
    param([string]$Text)
    Write-Host ""
    Write-Host $Text -ForegroundColor Cyan
}

Write-Host "=== grilly rebuild ===" -ForegroundColor Cyan
Write-Host "Root: $grillyRoot"

# ── 1. Compile GLSL -> SPIR-V ──────────────────────────────────────────────
if (-not $SkipShaders) {
    Write-Step "[1/3] Compiling shaders..."

    $shaderDir = Join-Path $grillyRoot "shaders"
    $spvDir    = Join-Path $shaderDir "spv"
    New-Item -ItemType Directory -Force -Path $spvDir | Out-Null

    # Verify glslangValidator is on PATH (ships with Vulkan SDK).
    $glslang = "glslangValidator"
    $glslangOk = $false
    try {
        $null = & $glslang --version 2>&1
        if ($LASTEXITCODE -eq 0) { $glslangOk = $true }
    } catch {}
    if (-not $glslangOk) {
        Write-Host "ERROR: glslangValidator not found on PATH." -ForegroundColor Red
        Write-Host "       Install the Vulkan SDK (https://vulkan.lunarg.com/) or" -ForegroundColor Red
        Write-Host "       add `$env:VULKAN_SDK/Bin to `$env:Path." -ForegroundColor Red
        exit 1
    }

    $total    = 0
    $compiled = 0
    $skipped  = 0
    $failed   = 0
    $failedNames = @()

    Get-ChildItem -Path $shaderDir -Filter "*.glsl" | ForEach-Object {
        $total++
        $src = $_.FullName
        $dst = Join-Path $spvDir ($_.BaseName + ".spv")

        # Skip if .spv is newer than .glsl
        if ((Test-Path $dst) -and ((Get-Item $dst).LastWriteTime -gt $_.LastWriteTime)) {
            $skipped++
            if ($VerbosePreference -eq 'Continue') {
                Write-Host "  SKIP $($_.BaseName) (up-to-date)" -ForegroundColor DarkGray
            }
            return
        }

        # -S comp  : treat as compute shader (glslangValidator can't infer
        #            the stage from the generic .glsl extension)
        # --target-env vulkan1.3 : enables cooperative matrix + subgroup extensions
        $output = & $glslang -V -S comp $src -o $dst --target-env vulkan1.3 2>&1
        if ($LASTEXITCODE -eq 0) {
            $compiled++
            if ($VerbosePreference -eq 'Continue') {
                Write-Host "  OK   $($_.BaseName)" -ForegroundColor DarkGreen
            }
        } else {
            $failed++
            $failedNames += $_.BaseName
            Write-Host "  FAIL $($_.BaseName)" -ForegroundColor Red
            $output | ForEach-Object { Write-Host "       $_" -ForegroundColor DarkRed }
        }
    }

    $summary = "  Shaders: $compiled compiled, $skipped up-to-date, $failed failed / $total total"
    if ($failed -gt 0) {
        Write-Host $summary -ForegroundColor Yellow
        Write-Host "ERROR: $failed shader(s) failed to compile: $($failedNames -join ', ')" -ForegroundColor Red
        exit 1
    } else {
        Write-Host $summary -ForegroundColor Green
    }
}

# ── 2. Build grilly_core (build2/Release) ────────────────────────────────
if (-not $SkipBuild) {
    Write-Step "[2/3] Building grilly_core (build2/Release)..."

    $buildDir = Join-Path $grillyRoot "build2"

    if ($Clean -and (Test-Path $buildDir)) {
        Write-Host "  Cleaning $buildDir..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $buildDir
    }

    if (-not (Test-Path (Join-Path $buildDir "CMakeCache.txt"))) {
        Write-Host "  Configuring cmake..."
        & cmake -B $buildDir -G "Visual Studio 17 2022" -A x64
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: cmake configure failed" -ForegroundColor Red
            exit 1
        }
    }

    & cmake --build $buildDir --config Release --target grilly_core
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: cmake build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Build OK" -ForegroundColor Green
}

# ── 3. Copy freshly built .pyd to grilly root ────────────────────────────
Write-Step "[3/3] Copying grilly_core.cp312-win_amd64.pyd..."

$builtPyd = Join-Path $grillyRoot "build2/Release/grilly_core.cp312-win_amd64.pyd"
$destPyd  = Join-Path $grillyRoot "grilly_core.cp312-win_amd64.pyd"

if (-not (Test-Path $builtPyd)) {
    Write-Host "ERROR: Built .pyd not found at $builtPyd" -ForegroundColor Red
    Write-Host "       (did the build succeed?)" -ForegroundColor Red
    exit 1
}

Copy-Item -Force $builtPyd $destPyd
$sizeMB = [math]::Round((Get-Item $destPyd).Length / 1MB, 2)
$mtime  = (Get-Item $destPyd).LastWriteTime.ToString("HH:mm:ss")
Write-Host "  Copied ($sizeMB MB, mtime $mtime)" -ForegroundColor Green

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
