# GPU Setup

Grilly uses Vulkan compute shaders for GPU acceleration. You need Vulkan drivers installed on your system.

---

## Windows

### 1. Install Visual Studio 2022

Download [Visual Studio 2022 Community](https://visualstudio.microsoft.com/) (free). In the installer, select **"Desktop development with C++"**. This installs MSVC, CMake, and Windows SDK.

### 2. Install the Vulkan SDK

Download from [https://vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home) (Windows tab). Run the installer -- it sets `VULKAN_SDK` automatically. Restart your terminal.

### 3. Verify

```powershell
vulkaninfo --summary
```

You should see your GPU listed with Vulkan 1.2+ support.

---

## Ubuntu 24.04

Ubuntu does **not** bundle the Vulkan SDK. Install from LunarG.

### 1. Add the LunarG APT Repository

```bash
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | \
    sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list \
    https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
sudo apt update
```

### 2. Install the Vulkan SDK

```bash
sudo apt install vulkan-sdk
```

### 3. Install GPU-Specific Drivers

=== "AMD (RADV)"

    ```bash
    sudo apt install mesa-vulkan-drivers
    ```

    RADV is the open-source Vulkan driver for AMD. Recommended for consumer cards.

=== "NVIDIA"

    ```bash
    sudo apt install nvidia-driver-560  # or latest version
    ```

=== "Intel Arc"

    ```bash
    sudo apt install intel-media-va-driver-non-free mesa-vulkan-drivers
    ```

### 4. Verify

```bash
vulkaninfo --summary
```

If your GPU is not listed, check that drivers are loaded:

```bash
lsmod | grep amdgpu   # AMD
lsmod | grep nvidia    # NVIDIA
```

### 5. Verify Grilly Sees the GPU

```bash
python -c "import grilly; backend = grilly.Compute(); print('OK:', backend.device_name)"
```

---

## CI / Headless Servers (No Physical GPU)

For CI environments without a physical GPU, use Mesa's software Vulkan driver:

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
export ALLOW_CPU_VULKAN=1
```

This is slow but runs the full test suite without GPU hardware.

---

## Multi-GPU Selection

If you have multiple GPUs (e.g., integrated + discrete), select by index:

```bash
export VK_GPU_INDEX=1  # Use second GPU
```

List available GPUs:

```bash
vulkaninfo --summary | grep "GPU"
```

---

## Supported Devices

See [SUPPORTED_DEVICES.md](https://github.com/grillcheese-ai/grilly/blob/main/SUPPORTED_DEVICES.md) for the full tested device list.

**Tested configurations:**

- AMD RX 6750 XT (Windows 11, Ubuntu 24.04)
- NVIDIA GTX 1060+ (Windows, Linux)
- Intel Arc A-series (Windows, Linux)
- Mesa llvmpipe (CI, headless)
