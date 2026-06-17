.PHONY: help install install-dev test test-cpu test-gpu test-cpp test-all \
       compile-shaders verify-shaders build clean publish publish-test \
       lint format check cpp-configure cpp-build cpp-install cpp all

# ── Tool paths ───────────────────────────────────────────────────────────
VENV_PYTHON := .venv/Scripts/python.exe
PYTEST      := $(VENV_PYTHON) -m pytest
GLSLC       := glslc
TWINE       := twine

# ── C++ build config ────────────────────────────────────────────────────
CMAKE_BUILD_DIR := build312
CMAKE_CONFIG    := Release
PYD_NAME        := $(shell ls $(CMAKE_BUILD_DIR)/$(CMAKE_CONFIG)/grilly_core.*.pyd 2>/dev/null | head -1)

help:
	@echo "Grilly Makefile Commands:"
	@echo ""
	@echo "  make all             Full build: C++ + copy .pyd + tests"
	@echo ""
	@echo "C++ Backend:"
	@echo "  make cpp-configure   CMake configure (build312/, venv Python 3.12)"
	@echo "  make cpp-build       Compile C++ backend"
	@echo "  make cpp-install     Copy .pyd to repo root"
	@echo "  make cpp             configure + build + install (no tests)"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run Python tests (via venv pytest)"
	@echo "  make test-cpp        Run C++ backend tests only"
	@echo "  make test-cpu        Run CPU-only tests (skip GPU)"
	@echo "  make test-gpu        Run GPU tests only"
	@echo "  make test-all        Run all tests (Python + C++)"
	@echo "  make test-coverage   Run tests with coverage report"
	@echo ""
	@echo "Shaders:"
	@echo "  make compile-shaders Compile all GLSL shaders to SPIR-V"
	@echo "  make verify-shaders  Verify shader compilation"
	@echo ""
	@echo "Building:"
	@echo "  make build           Build wheel and source distribution"
	@echo "  make clean           Remove Python build artifacts (keeps cmake dirs)"
	@echo ""
	@echo "Publishing:"
	@echo "  make publish-test    Publish to Test PyPI"
	@echo "  make publish         Publish to production PyPI"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            Run linters (ruff)"
	@echo "  make format          Format code (black, isort)"
	@echo "  make check           Lint + CPU tests"

# ── Full build ──────────────────────────────────────────────────────────
all: cpp test-all
	@echo "Build complete. All tests passed."

# ── C++ Backend ─────────────────────────────────────────────────────────
cpp-configure:
	@echo "Configuring CMake ($(CMAKE_BUILD_DIR))..."
	cmake -B $(CMAKE_BUILD_DIR) -DPYBIND11_FINDPYTHON=ON \
		-DPython_EXECUTABLE="$(VENV_PYTHON)" \
		-DCMAKE_BUILD_TYPE=$(CMAKE_CONFIG)

cpp-build:
	@echo "Building C++ backend..."
	cmake -B $(CMAKE_BUILD_DIR); cmake --build build312

cpp-install:
	@echo "Copying .pyd to repo root..."
	@PYD=$$(ls $(CMAKE_BUILD_DIR)/$(CMAKE_CONFIG)/grilly_core.*.pyd 2>/dev/null | head -1); \
	if [ -z "$$PYD" ]; then \
		echo "ERROR: No .pyd found in $(CMAKE_BUILD_DIR)/$(CMAKE_CONFIG)/"; \
		exit 1; \
	fi; \
	DEST="$$(basename $$PYD)"; \
	if [ -f "$$DEST" ]; then rm -f "$$DEST" 2>/dev/null || true; fi; \
	cp "$$PYD" . && echo "Installed: $$DEST" || \
		echo "WARN: Copy failed (file locked by running process?). Kill pytest first."

cpp: cpp-build cpp-install
	@echo "C++ backend ready."

# ── Installation ────────────────────────────────────────────────────────
install:
	$(VENV_PYTHON) -m pip install -e .

install-dev:
	$(VENV_PYTHON) -m pip install -e ".[dev]"

# ── Testing ─────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --ignore=tests/benchmark_snn_fashion_mnist.py

test-cpp:
	$(PYTEST) tests/ -v -k "cpp" -m "cpp"

test-cpu:
	$(PYTEST) tests/ -m "not gpu" -v

test-gpu:
	$(PYTEST) tests/ -k "gpu" -v

test-all:
	$(PYTEST) tests/ -v --ignore=tests/benchmark_snn_fashion_mnist.py

test-coverage:
	$(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term --cov-fail-under=40 -v
	@echo "Coverage report generated in htmlcov/index.html"

# ── Shaders ─────────────────────────────────────────────────────────────
compile-shaders:
ifeq ($(OS),Windows_NT)
	@echo "Compiling shaders on Windows..."
	powershell -ExecutionPolicy Bypass -File scripts/compile_all_shaders.ps1
else
	@echo "Compiling shaders on Unix..."
	bash compile_shaders.sh
endif

verify-shaders:
	@echo "Verifying shader compilation..."
	@GLSL_COUNT=$$(find shaders -name "*.glsl" -o -name "*.comp" 2>/dev/null | wc -l); \
	SPV_COUNT=$$(find shaders/spv -name "*.spv" 2>/dev/null | wc -l); \
	echo "GLSL source files: $$GLSL_COUNT"; \
	echo "Compiled SPIR-V files: $$SPV_COUNT"; \
	if [ $$SPV_COUNT -lt $$GLSL_COUNT ]; then \
		echo "WARNING: Not all shaders are compiled!"; \
		exit 1; \
	fi

# ── Python packaging ───────────────────────────────────────────────────
build:
	@echo "Building distribution packages..."
	$(VENV_PYTHON) -m build
	@echo "Build complete. Packages in dist/"

clean:
	@echo "Cleaning Python build artifacts..."
	rm -rf dist/ *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -maxdepth 3 -type d -name __pycache__ -not -path "*/build*" -not -path "*/third_party/*" -exec rm -rf {} + 2>/dev/null || true
	find . -maxdepth 3 -type f -name "*.pyc" -not -path "*/build*" -not -path "*/third_party/*" -delete 2>/dev/null || true
	@echo "Clean complete. (cmake dirs preserved — use 'rm -rf build312/' to remove)"

# ── Publishing ──────────────────────────────────────────────────────────
publish-test: build
	@echo "Publishing to Test PyPI..."
	$(TWINE) upload --repository testpypi dist/*
	@echo "Test PyPI upload complete."
	@echo "Install with: pip install --index-url https://test.pypi.org/simple/ grilly"

publish: build
	@echo "Publishing to PyPI..."
	@read -p "Are you sure you want to publish to production PyPI? (y/N) " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(TWINE) upload dist/*; \
		echo "PyPI upload complete."; \
		echo "Install with: pip install grilly"; \
	else \
		echo "Publish cancelled."; \
	fi

# ── Code quality ────────────────────────────────────────────────────────
lint:
	@echo "Running linters..."
	ruff check .
	@echo "Lint complete."

format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "Format complete."

check: lint test-cpu
	@echo "All checks passed."

.DEFAULT_GOAL := help
