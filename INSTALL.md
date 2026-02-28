# Installation Guide

> Currently tested on **Windows 11** and **Ubuntu 24.04**.

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.12+ | 3.12 |
| GPU VRAM | 8 GB | 12 GB+ |
| System RAM | 32 GB DDR4 | 64 GB DDR5 |
| CPU | Intel i5 9th gen / Ryzen 5 3600 | Intel i5 12th gen+ / Ryzen 7 5800+ |
| Vulkan | 1.2+ drivers | Latest drivers |

### Supported GPUs

Any GPU with Vulkan 1.2+ compute support:

- **AMD**: RX 5000 series and newer (tested on RX 6750 XT)
- **NVIDIA**: GTX 1060 and newer
- **Intel**: Arc A-series

See [SUPPORTED_DEVICES.md](./SUPPORTED_DEVICES.md) for the full tested device list.

---

## Quick Install (PyPI)

```bash
pip install grilly
```

This installs the Python package with all SPIR-V shaders pre-compiled. You still need Vulkan drivers installed on your system (see below).

## Install from Source

```bash
git clone https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"

# Build the C++ backend (required)
cmake -B build -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Release
cp build/Release/grilly_core.*.pyd .   # Windows
# cp build/grilly_core.*.so .          # Linux
```

### Verify Installation

```bash
python -c "import grilly_core; print('C++ backend OK')"
python -c "import grilly; backend = grilly.Compute(); print('GPU OK:', backend.device_name)"
```

---

## Installing Vulkan Support

### Windows

1. **Install Visual Studio 2022** (Community edition is free)
   - In the installer, select **"Desktop development with C++"** workload
   - This installs MSVC, CMake, and Windows SDK

2. **Install the Vulkan SDK**
   - Download from https://vulkan.lunarg.com/sdk/home (Windows tab)
   - Run the installer — it sets `VULKAN_SDK` environment variable automatically
   - Restart your terminal after installation

3. **Verify Vulkan**
   ```powershell
   vulkaninfo --summary
   ```

### Ubuntu 24.04

Ubuntu does **not** bundle the Vulkan SDK. You must install it manually from LunarG.

1. **Add the LunarG APT repository**
   ```bash
   wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
   sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
   sudo apt update
   ```

2. **Install the Vulkan SDK**
   ```bash
   sudo apt install vulkan-sdk
   ```

3. **Install GPU-specific Vulkan drivers**

   **AMD (RADV — open source, recommended):**
   ```bash
   sudo apt install mesa-vulkan-drivers
   ```

   **NVIDIA (proprietary):**
   ```bash
   sudo apt install nvidia-driver-560  # or latest version
   ```

   **Intel Arc:**
   ```bash
   sudo apt install intel-media-va-driver-non-free mesa-vulkan-drivers
   ```

4. **Verify Vulkan**
   ```bash
   vulkaninfo --summary
   ```
   You should see your GPU listed. If not, check that your drivers are loaded (`lsmod | grep amdgpu` or `lsmod | grep nvidia`).

5. **Verify grilly sees the GPU**
   ```bash
   python -c "import grilly; backend = grilly.Compute(); print('OK:', backend.device_name)"
   ```

### Ubuntu CI / Headless Servers (no physical GPU)

For CI environments or headless servers without a physical GPU, you can use the Mesa software Vulkan driver (llvmpipe):

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
export ALLOW_CPU_VULKAN=1
```

This is slow but allows running the full test suite without GPU hardware.

---

## C++ Backend (grilly_core)

The C++ backend (`grilly_core`) is the primary GPU dispatch layer. It provides a native pybind11 extension that wraps all Vulkan compute operations: linear, convolution, activations (ReLU, GELU, SiLU, tanh), normalization (LayerNorm, RMSNorm, BatchNorm), attention (Flash Attention 2, RoPE, KV-cache), embedding, pooling, loss functions, SNN neurons, and optimizers. **You must build this to use grilly.**

### Build Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| CMake | 3.20+ | Build system |
| Vulkan SDK | 1.2+ | Headers + loader |
| pybind11 | 2.11+ | Auto-fetched by CMake |
| Boost | 1.74+ | Optional, auto-fetched if missing |
| VMA | Latest | Auto-fetched by CMake |

### Build on Windows

```powershell
# From the grilly repo root
cmake -B build -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Release

# Copy the extension to the repo root
cp build/Release/grilly_core.cp312-win_amd64.pyd .
```

If using a venv instead of system Python:

```powershell
cmake -B build -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=".venv/Scripts/python.exe"
cmake --build build --config Release
cp build/Release/grilly_core.cp312-win_amd64.pyd .
```

### Build on Ubuntu

```bash
# Install build tools
sudo apt install cmake build-essential

# The Vulkan SDK (installed above) provides headers and the loader

# Build
cmake -B build -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Release -j$(nproc)

# Copy the extension
cp build/grilly_core.cpython-312-x86_64-linux-gnu.so .
```

### Verify C++ Backend

```bash
python -c "import grilly_core; print('C++ backend loaded:', dir(grilly_core))"
```

All `nn.Linear`, `nn.RMSNorm`, and other modules dispatch through `grilly_core` for GPU acceleration.

---

## optimum-grilly (HuggingFace Integration)

Run HuggingFace transformer models on Vulkan GPUs:

```bash
pip install optimum-grilly[gpu]
```

See the [optimum-grilly repository](https://github.com/grillcheese-ai/optimum-grilly) for full documentation.

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

- Check drivers are installed: `vulkaninfo --summary`
- On laptops with hybrid graphics, ensure the discrete GPU is active
- Try `export VK_GPU_INDEX=1` if you have integrated + discrete GPUs

### "SPIR-V shaders not found"

- If installing from source, shaders are in `shaders/spv/`
- Recompile shaders: `make compile-shaders` or `.\scripts\compile_all_shaders.ps1` (Windows)

### "grilly_core not found" (C++ backend)

- You must build the C++ backend — see the build instructions above
- Make sure the `.pyd` (Windows) or `.so` (Linux) file is in the repo root or on `sys.path`
- Check the build output for errors: `cmake --build build --config Release --verbose`

### CI test failures with "attempted relative import"

- Ensure grilly is installed in editable mode: `pip install -e .`
- If running without pip install, set `PYTHONPATH=.`
