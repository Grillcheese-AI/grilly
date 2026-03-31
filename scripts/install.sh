#!/usr/bin/env bash
# install.sh — One-shot grilly install with Vulkan GPU backend
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/Grillcheese-AI/grilly/main/scripts/install.sh | bash
#   # or locally:
#   ./scripts/install.sh
#   ./scripts/install.sh --fast    # Colab/CI: skip validation layers (~5 min vs ~30 min)
#
# Installs: system deps → Vulkan SDK 1.4 → builds grilly C++ → installs Python package
# Supports: Ubuntu/Debian (incl. Colab), macOS (via MoltenVK)
# Requires: Python 3.12+, cmake 3.20+, git

set -euo pipefail

VULKAN_SDK_VERSION="1.4.341.1"
GRILLY_DIR="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || echo "$(pwd)")"
BUILD_DIR="${GRILLY_DIR}/build"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
FAST_MODE=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --fast|-f) FAST_MODE=true ;;
        --help|-h) echo "Usage: $0 [--fast]"; echo "  --fast  Only build shaderc + loader (5 min vs 30 min)"; exit 0 ;;
    esac
done

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[grilly]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ── Detect platform ──────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="macos" ;;
    *)      fail "Unsupported platform: $OS (use WSL on Windows)" ;;
esac
info "Platform: $PLATFORM ($OS), jobs: $JOBS"

# ── Step 1: System dependencies ──────────────────────────────────────────────
info "Installing system dependencies..."
if [ "$PLATFORM" = "linux" ]; then
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            cmake g++ gcc git python3-dev pkg-config ninja-build \
            libxcb-dri3-0 libxcb-present0 libpciaccess0 libpng-dev \
            libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev \
            libwayland-dev libxrandr-dev libxcb-randr0-dev \
            libxcb-ewmh-dev libx11-xcb-dev liblz4-dev libzstd-dev \
            wayland-protocols xz-utils 2>/dev/null

        # Vulkan runtime + ICD (needed for GPU access on Colab / headless servers)
        sudo apt-get install -y -qq libvulkan1 libvulkan-dev vulkan-tools 2>/dev/null

        # Colab/cloud: install NVIDIA Vulkan ICD if NVIDIA GPU detected
        if [ -d /proc/driver/nvidia ] || command -v nvidia-smi &>/dev/null; then
            info "NVIDIA GPU detected — installing Vulkan ICD driver..."
            sudo apt-get install -y -qq nvidia-driver-550 2>/dev/null || \
            sudo apt-get install -y -qq nvidia-drivers-550 2>/dev/null || \
                warn "Could not install nvidia-driver-550 — Vulkan may not see GPU"
        fi

        ok "System deps installed (apt)"
    else
        warn "No apt-get found — assuming deps are installed"
    fi
elif [ "$PLATFORM" = "macos" ]; then
    if command -v brew &>/dev/null; then
        brew install cmake ninja pkg-config 2>/dev/null || true
        ok "System deps installed (brew)"
    else
        warn "No Homebrew — install cmake and ninja manually"
    fi
fi

# ── Step 2: Vulkan SDK ───────────────────────────────────────────────────────
install_vulkan_linux() {
    if [ -n "${VULKAN_SDK:-}" ] && [ -f "${VULKAN_SDK}/bin/glslangValidator" ]; then
        ok "Vulkan SDK already installed at $VULKAN_SDK"
        return
    fi

    local SDK_DIR="/opt/vulkan/${VULKAN_SDK_VERSION}"
    if [ -f "${SDK_DIR}/x86_64/bin/glslangValidator" ]; then
        export VULKAN_SDK="${SDK_DIR}/x86_64"
        ok "Vulkan SDK found at $VULKAN_SDK"
        return
    fi

    info "Downloading Vulkan SDK ${VULKAN_SDK_VERSION}..."
    local TMP_DIR=$(mktemp -d)
    local TAR="vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"
    wget -q "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/${TAR}" \
        -O "${TMP_DIR}/${TAR}" || fail "Failed to download Vulkan SDK"

    info "Extracting Vulkan SDK..."
    sudo mkdir -p /opt/vulkan
    sudo tar xf "${TMP_DIR}/${TAR}" -C /opt/vulkan/
    rm -rf "$TMP_DIR"

    export VULKAN_SDK="/opt/vulkan/${VULKAN_SDK_VERSION}/x86_64"

    if [ -f "/opt/vulkan/${VULKAN_SDK_VERSION}/vulkansdk" ]; then
        cd "/opt/vulkan/${VULKAN_SDK_VERSION}"
        if [ "$FAST_MODE" = true ]; then
            info "Building Vulkan SDK (fast mode: shaderc + loader only, ~5 min)..."
            sudo bash ./vulkansdk -j "$JOBS" vulkan-headers vulkan-loader shaderc 2>&1 | tail -5
        else
            info "Building Vulkan SDK (full: all components, ~30 min)..."
            info "  Tip: use --fast to skip validation layers and build in ~5 min"
            sudo bash ./vulkansdk all -j "$JOBS" 2>&1 | tail -5
        fi
        cd "$GRILLY_DIR"
    fi

    ok "Vulkan SDK ${VULKAN_SDK_VERSION} installed at $VULKAN_SDK"
}

install_vulkan_macos() {
    if [ -n "${VULKAN_SDK:-}" ] && [ -d "$VULKAN_SDK" ]; then
        ok "Vulkan SDK already set: $VULKAN_SDK"
        return
    fi

    # Check common MoltenVK / LunarG install paths
    for p in "$HOME/VulkanSDK/"*/macOS /usr/local/share/vulkan; do
        if [ -d "$p" ]; then
            export VULKAN_SDK="$p"
            ok "Vulkan SDK found at $VULKAN_SDK"
            return
        fi
    done

    warn "Vulkan SDK not found on macOS."
    warn "Download from: https://vulkan.lunarg.com/sdk/home#mac"
    warn "Building grilly without Vulkan (CPU-only fallback)..."
}

info "Setting up Vulkan SDK..."
if [ "$PLATFORM" = "linux" ]; then
    install_vulkan_linux
elif [ "$PLATFORM" = "macos" ]; then
    install_vulkan_macos
fi

# Export paths
if [ -n "${VULKAN_SDK:-}" ]; then
    export PATH="${VULKAN_SDK}/bin:$PATH"
    export LD_LIBRARY_PATH="${VULKAN_SDK}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PKG_CONFIG_PATH="${VULKAN_SDK}/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export CMAKE_PREFIX_PATH="${VULKAN_SDK}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
fi

# ── Step 3: Clone grilly if not in repo ──────────────────────────────────────
if [ ! -f "${GRILLY_DIR}/CMakeLists.txt" ]; then
    info "Cloning grilly..."
    GRILLY_DIR="$(pwd)/grilly"
    git clone --recurse-submodules https://github.com/Grillcheese-AI/grilly.git "$GRILLY_DIR"
fi

# Update BUILD_DIR now that GRILLY_DIR is resolved
BUILD_DIR="${GRILLY_DIR}/build"
cd "$GRILLY_DIR"
info "Source directory: $GRILLY_DIR"

# ── Step 4: Build C++ extension ──────────────────────────────────────────────
info "Building grilly C++ extension (Vulkan compute backend)..."
mkdir -p "$BUILD_DIR"

cmake -S "$GRILLY_DIR" -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(python3 -c 'import sys; print(sys.executable)')" \
    ${VULKAN_SDK:+-DVulkan_INCLUDE_DIR="${VULKAN_SDK}/include"} \
    ${VULKAN_SDK:+-DVulkan_LIBRARY="${VULKAN_SDK}/lib/libvulkan.so"} \
    2>&1 | tail -10

cmake --build "$BUILD_DIR" --parallel "$JOBS" 2>&1 | tail -10

# Find the built .so/.pyd
BUILT_LIB=$(find "$BUILD_DIR" -name "grilly_core*.so" -o -name "grilly_core*.pyd" 2>/dev/null | head -1)
if [ -z "$BUILT_LIB" ]; then
    # Try _bridge naming
    BUILT_LIB=$(find "$BUILD_DIR" -name "_bridge*.so" -o -name "_bridge*.pyd" 2>/dev/null | head -1)
fi

if [ -n "$BUILT_LIB" ]; then
    ok "C++ extension built: $BUILT_LIB"
else
    warn "C++ extension not found in build output — check build logs"
fi

cd "$GRILLY_DIR"

# ── Step 5: Install Python package ───────────────────────────────────────────
info "Installing grilly Python package..."
pip install -e "." 2>&1 | tail -3

# Copy the built C++ extension into the package
if [ -n "${BUILT_LIB:-}" ]; then
    SITE_PKG=$(python3 -c "import grilly; print(grilly.__path__[0])")
    cp "$BUILT_LIB" "$SITE_PKG/"
    ok "C++ extension copied to $SITE_PKG"
fi

# ── Step 6: Verify ───────────────────────────────────────────────────────────
info "Verifying installation..."
python3 -c "
import grilly
print(f'  grilly {getattr(grilly, \"__version__\", \"?\")}'  )
try:
    from grilly.backend import _bridge
    if _bridge.is_available():
        print('  Vulkan backend: ENABLED')
    else:
        print('  Vulkan backend: loaded but not available')
except Exception as e:
    print(f'  Vulkan backend: not available ({e})')
print('  NumPy fallback: always available')
"

echo ""
ok "grilly installed successfully!"
