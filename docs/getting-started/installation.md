# Installation

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.12+ | 3.12 |
| GPU VRAM | 8 GB | 12 GB+ |
| System RAM | 32 GB | 64 GB |
| Vulkan | 1.2+ | Latest drivers |

**Supported GPUs**: AMD (RX 5000+), NVIDIA (GTX 1060+), Intel (Arc A-series).

---

## Quick Install (PyPI)

```bash
pip install grilly
```

This installs the Python package with all SPIR-V shaders pre-compiled. You still need Vulkan drivers on your system (see [GPU Setup](gpu-setup.md)).

Without the C++ extension, grilly works fully via pure Python + numpy fallbacks -- just without GPU acceleration.

---

## Install from Source

```bash
git clone https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"
```

### Build the C++ Backend

The C++ backend (`grilly_core`) is a pybind11 extension that wraps all Vulkan compute operations. It provides zero-copy GPU dispatch with VMA persistent mapping and BufferPool allocation.

**Build requirements:**

| Dependency | Version | Notes |
|-----------|---------|-------|
| CMake | 3.20+ | Build system |
| Vulkan SDK | 1.2+ | Headers + loader |
| pybind11 | 2.11+ | Auto-fetched by CMake |
| VMA | Latest | Auto-fetched by CMake |

=== "Windows"

    ```powershell
    cmake -B build -DPYBIND11_FINDPYTHON=ON
    cmake --build build --config Release
    cp build/Release/grilly_core.cp312-win_amd64.pyd .
    ```

    Requires Visual Studio 2022 with the "Desktop development with C++" workload.

    If using a venv:

    ```powershell
    cmake -B build -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=".venv/Scripts/python.exe"
    cmake --build build --config Release
    cp build/Release/grilly_core.cp312-win_amd64.pyd .
    ```

=== "Linux"

    ```bash
    sudo apt install cmake build-essential
    cmake -B build -DPYBIND11_FINDPYTHON=ON
    cmake --build build --config Release -j$(nproc)
    cp build/grilly_core.cpython-312-x86_64-linux-gnu.so .
    ```

### Pre-Built Binary (Windows x64)

Download `grilly_core.cp312-win_amd64.pyd` from the [latest release](https://github.com/grillcheese-ai/grilly/releases) and place it in your grilly install directory:

```bash
python -c "import grilly; print(grilly.__file__)"
# Copy the .pyd to that directory
```

---

## Verify Installation

```bash
# Check Python package
python -c "import grilly; print('grilly', grilly.__version__)"

# Check C++ backend
python -c "import grilly_core; print('C++ backend OK')"

# Check GPU access
python -c "import grilly; backend = grilly.Compute(); print('GPU:', backend.device_name)"
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VK_GPU_INDEX` | Select GPU by index (multi-GPU systems) | `0` |
| `GRILLY_DEBUG` | Enable debug logging (`1` = on) | off |
| `ALLOW_CPU_VULKAN` | Allow Mesa llvmpipe software Vulkan | off |

---

## Troubleshooting

### "No Vulkan devices found"

- Check drivers: `vulkaninfo --summary`
- On laptops with hybrid graphics, ensure the discrete GPU is active
- Try `VK_GPU_INDEX=1` if you have integrated + discrete GPUs

### "SPIR-V shaders not found"

- If installing from source, shaders are in `shaders/spv/`
- Recompile: `.\scripts\compile_all_shaders.ps1` (Windows)

### "grilly_core not found"

- Build the C++ backend (see above)
- Ensure the `.pyd` / `.so` is in the repo root or on `sys.path`
- Check build output: `cmake --build build --config Release --verbose`

### CI test failures with "attempted relative import"

- Install in editable mode: `pip install -e .`
- Or set `PYTHONPATH=.`
