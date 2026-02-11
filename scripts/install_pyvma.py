#!/usr/bin/env python3
"""
PyVMA Installation Script

Automates the installation of PyVMA (Python wrapper for Vulkan Memory Allocator).

PyVMA provides AMD/NVIDIA/Intel optimized GPU memory allocation for Grilly.

Usage:
    python -m grilly.scripts.install_pyvma

    Or from command line:
    python grilly/scripts/install_pyvma.py

Requirements:
    - Windows: Visual Studio Build Tools with C++ workload
    - Linux: GCC/G++ and development headers
    - Vulkan SDK (optional, for headers)
"""

import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

# VMA release URL (header-only library)
VMA_VERSION = "3.0.1"
VMA_HEADER_URL = f"https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/v{VMA_VERSION}/include/vk_mem_alloc.h"

# PyVMA repository
PYVMA_REPO = "https://github.com/realitix/pyvma.git"


def print_step(msg: str):
    """Print a step message"""
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}\n")


def print_error(msg: str):
    """Print an error message"""
    print(f"\n[ERROR] {msg}", file=sys.stderr)


def print_success(msg: str):
    """Print a success message"""
    print(f"\n[SUCCESS] {msg}")


def check_compiler() -> bool:
    """Check if a C++ compiler is available"""
    if platform.system() == "Windows":
        # Check for MSVC
        result = subprocess.run(["where", "cl.exe"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Found MSVC compiler: cl.exe")
            return True

        # Try to find VS Developer Command Prompt
        vs_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        ]
        for vs_path in vs_paths:
            if os.path.exists(vs_path):
                print(f"Found Visual Studio at: {vs_path}")
                return True

        print_error("MSVC compiler not found. Install Visual Studio Build Tools with C++ workload.")
        print("Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        return False
    else:
        # Check for GCC/G++
        result = subprocess.run(["which", "g++"], capture_output=True)
        if result.returncode == 0:
            print("Found G++ compiler")
            return True
        print_error("G++ compiler not found. Install build-essential or equivalent.")
        return False


def download_vma_header(target_dir: Path) -> bool:
    """Download VMA header file"""
    target_file = target_dir / "vk_mem_alloc.h"

    if target_file.exists():
        print(f"VMA header already exists: {target_file}")
        return True

    print("Downloading VMA header from GitHub...")
    try:
        urllib.request.urlretrieve(VMA_HEADER_URL, target_file)
        print(f"Downloaded: {target_file}")
        return True
    except Exception as e:
        print_error(f"Failed to download VMA header: {e}")
        return False


def build_vma_lib_windows(pyvma_dir: Path) -> bool:
    """Build vk_mem_alloc.lib on Windows"""
    build_dir = pyvma_dir / "pyvma" / "pyvma_build"

    if not build_dir.exists():
        print_error(f"PyVMA build directory not found: {build_dir}")
        return False

    # Check if lib already exists
    lib_file = build_dir / "vk_mem_alloc.lib"
    if lib_file.exists():
        print(f"VMA library already exists: {lib_file}")
        return True

    # Download VMA header if needed
    if not download_vma_header(build_dir):
        return False

    header_file = build_dir / "vk_mem_alloc.h"
    obj_file = build_dir / "vk_mem_alloc.obj"
    include_dir = build_dir / "include"

    # Compile VMA
    print("Compiling VMA library...")

    # Find vcvars64.bat
    vcvars_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]

    vcvars = None
    for path in vcvars_paths:
        if os.path.exists(path):
            vcvars = path
            break

    if vcvars is None:
        print_error("Could not find vcvars64.bat")
        return False

    # Create batch script to compile
    compile_script = f'''
@echo off
call "{vcvars}"
cd /d "{build_dir}"
cl.exe /c /I"{include_dir}" /DVMA_IMPLEMENTATION /DVMA_STATIC_VULKAN_FUNCTIONS=0 /nologo /W3 /Ox /Oi /GF /EHsc /MD /GS /Gy /Zc:inline /Zc:wchar_t /Gd /TP /Fo"{obj_file}" "{header_file}"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
lib.exe /OUT:"{lib_file}" "{obj_file}"
'''

    script_file = build_dir / "compile_vma.bat"
    with open(script_file, "w") as f:
        f.write(compile_script)

    result = subprocess.run(["cmd", "/c", str(script_file)], capture_output=True, text=True)

    if result.returncode != 0:
        print_error(f"Compilation failed:\n{result.stderr}")
        return False

    if lib_file.exists():
        print_success(f"Built VMA library: {lib_file}")
        return True
    else:
        print_error("VMA library was not created")
        return False


def build_vma_lib_linux(pyvma_dir: Path) -> bool:
    """Build libvk_mem_alloc.a on Linux"""
    build_dir = pyvma_dir / "pyvma" / "pyvma_build"

    if not build_dir.exists():
        print_error(f"PyVMA build directory not found: {build_dir}")
        return False

    lib_file = build_dir / "libvk_mem_alloc.a"
    if lib_file.exists():
        print(f"VMA library already exists: {lib_file}")
        return True

    if not download_vma_header(build_dir):
        return False

    header_file = build_dir / "vk_mem_alloc.h"
    obj_file = build_dir / "vk_mem_alloc.o"
    include_dir = build_dir / "include"

    print("Compiling VMA library...")

    # Compile
    cmd1 = [
        "g++",
        "-std=c++11",
        "-fPIC",
        "-x",
        "c++",
        f"-I{include_dir}",
        "-DVMA_IMPLEMENTATION",
        "-DVMA_STATIC_VULKAN_FUNCTIONS=0",
        "-c",
        str(header_file),
        "-o",
        str(obj_file),
    ]

    result = subprocess.run(cmd1, capture_output=True, text=True)
    if result.returncode != 0:
        print_error(f"Compilation failed:\n{result.stderr}")
        return False

    # Create static library
    cmd2 = ["ar", "rvs", str(lib_file), str(obj_file)]
    result = subprocess.run(cmd2, capture_output=True, text=True)

    if result.returncode != 0:
        print_error(f"Library creation failed:\n{result.stderr}")
        return False

    print_success(f"Built VMA library: {lib_file}")
    return True


def clone_pyvma(target_dir: Path) -> bool:
    """Clone PyVMA repository"""
    if (target_dir / "setup.py").exists():
        print(f"PyVMA already cloned: {target_dir}")
        return True

    print("Cloning PyVMA repository...")
    result = subprocess.run(
        ["git", "clone", PYVMA_REPO, str(target_dir)], capture_output=True, text=True
    )

    if result.returncode != 0:
        print_error(f"Failed to clone PyVMA:\n{result.stderr}")
        return False

    print_success(f"Cloned PyVMA to: {target_dir}")
    return True


def fix_pyvma_paths(pyvma_dir: Path):
    """Fix path handling in PyVMA setup.py for paths with spaces"""
    setup_file = pyvma_dir / "setup.py"

    if not setup_file.exists():
        return

    with open(setup_file) as f:
        content = f.read()

    # Check if already fixed
    if "Quote paths" in content:
        print("PyVMA setup.py already has path fixes")
        return

    # Fix Windows build command
    old_windows = """    def build_windows(self):
        p1 = path.join(self.p, 'include')
        p2 = path.join(self.p, 'vk_mem_alloc.h')
        p3 = path.join(self.p, 'vk_mem_alloc.obj')
        p4 = path.join(self.p,  'vk_mem_alloc.lib')
        c = ' /DVMA_IMPLEMENTATION /DVMA_STATIC_VULKAN_FUNCTIONS=0 '
        cmd1 = 'cl.exe /c /I' + p1 + c +' /nologo /W3 /WX /Ox /Oi /GF /Gm- /EHsc /MD /GS /Gy /Zc:inline /Zc:wchar_t /Gd /TP /errorReport:none /Fo' + p3 + ' ' + p2  # noqa

        cmd2 = 'lib.exe /OUT:' + p4 + ' ' + p3

        call(cmd1, shell=True)
        call(cmd2, shell=True)"""

    new_windows = """    def build_windows(self):
        p1 = path.join(self.p, 'include')
        p2 = path.join(self.p, 'vk_mem_alloc.h')
        p3 = path.join(self.p, 'vk_mem_alloc.obj')
        p4 = path.join(self.p, 'vk_mem_alloc.lib')
        c = ' /DVMA_IMPLEMENTATION /DVMA_STATIC_VULKAN_FUNCTIONS=0 '
        # Quote paths to handle spaces in directory names
        cmd1 = 'cl.exe /c /I"' + p1 + '"' + c + ' /nologo /W3 /WX /Ox /Oi /GF /Gm- /EHsc /MD /GS /Gy /Zc:inline /Zc:wchar_t /Gd /TP /errorReport:none /Fo"' + p3 + '" "' + p2 + '"'  # noqa

        cmd2 = 'lib.exe /OUT:"' + p4 + '" "' + p3 + '"'

        call(cmd1, shell=True)
        call(cmd2, shell=True)"""

    content = content.replace(old_windows, new_windows)

    # Fix Linux build command
    old_linux = """    def build_linux(self):
        p1 = path.join(self.p, 'include')
        p2 = path.join(self.p, 'vk_mem_alloc.h')
        p3 = path.join(self.p, 'vk_mem_alloc.o')
        p4 = path.join(self.p, 'libvk_mem_alloc.a')
        c = ' -DVMA_IMPLEMENTATION -D_DEBUG -DVMA_STATIC_VULKAN_FUNCTIONS=0 '
        cmd1 = 'g++ -std=c++11 -fPIC -x c++ -I' + p1 + c + '-c ' + p2 + ' -o ' + p3  # noqa
        cmd2 = 'ar rvs ' + p4 + ' ' + p3

        call(cmd1, shell=True)
        call(cmd2, shell=True)"""

    new_linux = """    def build_linux(self):
        p1 = path.join(self.p, 'include')
        p2 = path.join(self.p, 'vk_mem_alloc.h')
        p3 = path.join(self.p, 'vk_mem_alloc.o')
        p4 = path.join(self.p, 'libvk_mem_alloc.a')
        c = ' -DVMA_IMPLEMENTATION -D_DEBUG -DVMA_STATIC_VULKAN_FUNCTIONS=0 '
        # Quote paths to handle spaces in directory names
        cmd1 = 'g++ -std=c++11 -fPIC -x c++ -I"' + p1 + '"' + c + '-c "' + p2 + '" -o "' + p3 + '"'  # noqa
        cmd2 = 'ar rvs "' + p4 + '" "' + p3 + '"'

        call(cmd1, shell=True)
        call(cmd2, shell=True)"""

    content = content.replace(old_linux, new_linux)

    # Fix build() to use absolute path
    old_build = """    def build(self):
        self.p = '.'  # `path.dirname(path.realpath(__file__))`
        self.p = path.join(self.p, 'pyvma')
        self.p = path.join(self.p, 'pyvma_build')"""

    new_build = """    def build(self):
        # Use absolute path to handle spaces in directory names
        self.p = path.dirname(path.realpath(__file__))
        self.p = path.join(self.p, 'pyvma')
        self.p = path.join(self.p, 'pyvma_build')"""

    content = content.replace(old_build, new_build)

    with open(setup_file, "w") as f:
        f.write(content)

    print("Fixed path handling in PyVMA setup.py")


def install_pyvma(pyvma_dir: Path) -> bool:
    """Install PyVMA package"""
    print("Installing PyVMA...")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", str(pyvma_dir)], capture_output=True, text=True
    )

    if result.returncode != 0:
        print_error(f"Failed to install PyVMA:\n{result.stderr}")
        return False

    print_success("PyVMA installed successfully!")
    return True


def verify_installation() -> bool:
    """Verify PyVMA installation"""
    print("Verifying installation...")

    try:
        import pyvma

        if hasattr(pyvma, "vmaCreateAllocator"):
            print_success("PyVMA is working correctly!")
            print(f"  Version: {getattr(pyvma, '__version__', 'unknown')}")
            print("  vmaCreateAllocator: available")
            print("  vmaCreateBuffer: available")
            return True
        else:
            print_error("PyVMA imported but VMA functions not available")
            return False
    except ImportError as e:
        print_error(f"Failed to import PyVMA: {e}")
        return False


def main():
    """Main installation procedure"""
    print_step("PyVMA Installation for Grilly")

    # Check compiler
    print_step("Step 1: Checking compiler")
    if not check_compiler():
        sys.exit(1)

    # Determine pyvma directory
    script_dir = Path(__file__).parent.parent.parent  # grilly/grilly -> grilly
    pyvma_dir = script_dir / "pyvma"

    # Clone if not exists
    print_step("Step 2: Getting PyVMA source")
    if not pyvma_dir.exists():
        if not clone_pyvma(pyvma_dir):
            sys.exit(1)
    else:
        print(f"Using existing PyVMA directory: {pyvma_dir}")

    # Fix paths in setup.py
    print_step("Step 3: Fixing path handling")
    fix_pyvma_paths(pyvma_dir)

    # Build VMA library
    print_step("Step 4: Building VMA library")
    if platform.system() == "Windows":
        if not build_vma_lib_windows(pyvma_dir):
            sys.exit(1)
    else:
        if not build_vma_lib_linux(pyvma_dir):
            sys.exit(1)

    # Install PyVMA
    print_step("Step 5: Installing PyVMA")
    if not install_pyvma(pyvma_dir):
        sys.exit(1)

    # Verify
    print_step("Step 6: Verifying installation")
    if not verify_installation():
        sys.exit(1)

    print_step("Installation Complete!")
    print("""
PyVMA is now installed and ready for use with Grilly.

To use VMA buffer pooling in your code:
    from grilly.backend.buffer_pool import get_buffer_pool, is_vma_available

    print(f"VMA available: {is_vma_available()}")
    pool = get_buffer_pool(vulkan_core)  # Uses VMA automatically
    buffer = pool.acquire(1024)
    # ... use buffer ...
    buffer.release()

For more information:
    https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
""")


if __name__ == "__main__":
    main()
