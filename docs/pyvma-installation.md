# PyVMA Installation Guide

PyVMA (Python Vulkan Memory Allocator) provides AMD/NVIDIA/Intel optimized GPU memory allocation for Grilly.

## Why PyVMA?

The default Vulkan memory allocation has known issues with AMD GPUs, causing memory access violations when buffers are reused. VMA (Vulkan Memory Allocator) from AMD's GPUOpen solves this by:

- **Sub-allocation**: Allocates from large memory blocks, reducing fragmentation
- **Optimal memory type selection**: Automatically chooses the best memory heap for your GPU
- **Persistent mapping**: Efficient CPU<->GPU transfers without repeated map/unmap
- **Defragmentation**: Handles memory fragmentation automatically

## Automatic Installation

The easiest way to install PyVMA:

```bash
python -m grilly.scripts.install_pyvma
```

This script will:
1. Check for a C++ compiler (MSVC on Windows, GCC on Linux)
2. Clone the PyVMA repository if needed
3. Build the VMA static library
4. Install PyVMA
5. Verify the installation

## Manual Installation

### Prerequisites

#### Windows
- **Visual Studio Build Tools** with C++ workload
  - Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - Install "Desktop development with C++" workload

#### Linux
```bash
sudo apt-get install build-essential  # Debian/Ubuntu
sudo dnf groupinstall "Development Tools"  # Fedora
```

### Step 1: Get PyVMA Source

```bash
cd grilly
git clone https://github.com/realitix/pyvma.git
```

### Step 2: Build VMA Library

#### Windows (from Developer Command Prompt)

```cmd
cd pyvma\pyvma\pyvma_build

REM Compile VMA
cl.exe /c /I"include" /DVMA_IMPLEMENTATION /DVMA_STATIC_VULKAN_FUNCTIONS=0 ^
    /nologo /W3 /Ox /Oi /GF /EHsc /MD /GS /Gy /Zc:inline /Zc:wchar_t /Gd /TP ^
    /Fo"vk_mem_alloc.obj" "vk_mem_alloc.h"

REM Create static library
lib.exe /OUT:"vk_mem_alloc.lib" "vk_mem_alloc.obj"
```

#### Linux

```bash
cd pyvma/pyvma/pyvma_build

# Compile VMA
g++ -std=c++11 -fPIC -x c++ -I"include" \
    -DVMA_IMPLEMENTATION -DVMA_STATIC_VULKAN_FUNCTIONS=0 \
    -c vk_mem_alloc.h -o vk_mem_alloc.o

# Create static library
ar rvs libvk_mem_alloc.a vk_mem_alloc.o
```

### Step 3: Fix Path Issues (for paths with spaces)

If your project path contains spaces, edit `pyvma/setup.py`:

1. In `build_windows()`, quote all paths:
```python
cmd1 = 'cl.exe /c /I"' + p1 + '"' + c + ' ... /Fo"' + p3 + '" "' + p2 + '"'
cmd2 = 'lib.exe /OUT:"' + p4 + '" "' + p3 + '"'
```

2. In `build()`, use absolute path:
```python
self.p = path.dirname(path.realpath(__file__))
```

### Step 4: Install PyVMA

```bash
cd pyvma
pip install .
```

### Step 5: Verify Installation

```python
import pyvma
print(pyvma.__version__)
print(pyvma.vmaCreateAllocator)  # Should print function reference
```

## Using PyVMA with Grilly

Once installed, Grilly automatically uses VMA for buffer allocation:

```python
from grilly import Compute
from grilly.backend.buffer_pool import is_vma_available, get_buffer_pool

# Check if VMA is available
print(f"VMA available: {is_vma_available()}")

# Initialize compute (VMA pool is created automatically)
compute = Compute()

# Get buffer pool stats
pool = get_buffer_pool(compute.core)
print(pool)  # VMABufferPool(pooled=0KB, hit_rate=0.0%, allocs=0, vma=True)
```

## Troubleshooting

### "cl.exe not found" (Windows)

Open "Developer Command Prompt for VS 2022" instead of regular Command Prompt:
- Start Menu → Visual Studio 2022 → Developer Command Prompt

Or manually initialize:
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

### "vk_mem_alloc.lib not found"

The VMA library wasn't built. Follow Step 2 above to build it manually.

### "cannot import name 'vma' from 'pyvma'"

The `__init__.py` might be empty. Check that PyVMA was built with the extension:
```python
from pyvma import _pyvma
print(dir(_pyvma.lib))  # Should list VMA functions
```

### Path with spaces causes build failure

Make sure to quote all paths in commands. The automatic installer handles this.

## Performance Benefits

With VMA enabled, you should see:
- Reduced memory allocation overhead
- Better buffer reuse (higher hit rate)
- No memory access violations on AMD GPUs
- Improved overall GPU performance

## References

- [VMA Documentation](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/)
- [VMA GitHub](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- [PyVMA GitHub](https://github.com/realitix/pyvma)
