# Installation Guide

> Tested on **Windows 11** and **Ubuntu 24.04**. For full documentation, see the [docs site](https://grillcheese-ai.github.io/grilly).

---

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.12+ | 3.12 |
| GPU VRAM | 8 GB | 12 GB+ |
| System RAM | 32 GB | 64 GB |
| Vulkan | 1.2+ drivers | Latest drivers |

**Supported GPUs**: AMD (RX 5000+), NVIDIA (GTX 1060+), Intel (Arc A-series). See [SUPPORTED_DEVICES.md](./SUPPORTED_DEVICES.md).

---

## Quick Install (PyPI)

```bash
pip install grilly
```

Installs the Python package with pre-compiled SPIR-V shaders. You still need Vulkan drivers (see [Vulkan Setup](#vulkan-setup) below).

Without the C++ extension, grilly works via pure Python + numpy fallbacks -- without GPU acceleration.

---

## Install from Source

```bash
git clone https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"
```

### Build the C++ Backend (grilly_core)

The C++ backend provides GPU-accelerated dispatch via pybind11, VMA persistent mapping, and BufferPool allocation.

**Build requirements**: CMake 3.20+, Vulkan SDK 1.2+, C++ compiler (MSVC or GCC). pybind11 and VMA are auto-fetched by CMake.

#### Windows

```powershell
cmake -B build -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Release
cp build/Release/grilly_core.cp312-win_amd64.pyd .
```

Requires Visual Studio 2022 with "Desktop development with C++". If using a venv:

```powershell
cmake -B build -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=".venv/Scripts/python.exe"
cmake --build build --config Release
cp build/Release/grilly_core.cp312-win_amd64.pyd .
```

#### Ubuntu

```bash
sudo apt install cmake build-essential
cmake -B build -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Release -j$(nproc)
cp build/grilly_core.cpython-312-x86_64-linux-gnu.so .
```

#### Pre-Built Binary (Windows x64)

Download `grilly_core.cp312-win_amd64.pyd` from the [latest release](https://github.com/grillcheese-ai/grilly/releases) and place it in your grilly install directory:

```bash
python -c "import grilly; print(grilly.__file__)"
# Copy the .pyd to that directory
```

---

## Verify Installation

```bash
python -c "import grilly; print('grilly', grilly.__version__)"
python -c "import grilly_core; print('C++ backend OK')"
python -c "import grilly; backend = grilly.Compute(); print('GPU:', backend.device_name)"
```

---

## Vulkan Setup

### Windows

1. Install [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (Windows tab)
2. Run the installer -- sets `VULKAN_SDK` automatically
3. Restart terminal
4. Verify: `vulkaninfo --summary`

### Ubuntu 24.04

```bash
# Add LunarG repo
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | \
    sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list \
    https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
sudo apt update

# Install SDK
sudo apt install vulkan-sdk

# Install GPU drivers
sudo apt install mesa-vulkan-drivers          # AMD (RADV)
# sudo apt install nvidia-driver-560          # NVIDIA
# sudo apt install intel-media-va-driver-non-free mesa-vulkan-drivers  # Intel

# Verify
vulkaninfo --summary
```

### CI / Headless (No GPU)

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
export ALLOW_CPU_VULKAN=1
```

Slow but runs the full test suite without hardware GPU.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VK_GPU_INDEX` | Select GPU by index | `0` |
| `GRILLY_DEBUG` | Debug logging (`1` = on) | off |
| `ALLOW_CPU_VULKAN` | Allow Mesa llvmpipe | off |

---

## Ecosystem

```bash
pip install optimum-grilly[gpu]  # HuggingFace Optimum backend for Vulkan inference
pip install cubemind             # Neuro-vector-symbolic reasoning on grilly
```

---

## Troubleshooting

**"No Vulkan devices found"**: Check drivers with `vulkaninfo --summary`. On hybrid GPU laptops, ensure the discrete GPU is active. Try `VK_GPU_INDEX=1`.

**"SPIR-V shaders not found"**: Shaders are in `shaders/spv/`. Recompile with `.\scripts\compile_all_shaders.ps1` (Windows).

**"grilly_core not found"**: Build the C++ backend (see above). Ensure `.pyd`/`.so` is in the repo root or on `sys.path`. Debug: `cmake --build build --config Release --verbose`.

**"attempted relative import"**: Install in editable mode: `pip install -e .` or set `PYTHONPATH=.`.
