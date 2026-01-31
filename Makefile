.PHONY: help install install-dev test test-cpu test-gpu compile-shaders build clean publish publish-test lint format check

PYTHON := python
PYTEST := pytest
GLSLC := glslc
TWINE := twine

help:
	@echo "Grilly Makefile Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install         Install package and dependencies"
	@echo "  make install-dev     Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-cpu       Run CPU-only tests (skip GPU)"
	@echo "  make test-gpu       Run GPU tests only"
	@echo "  make test-coverage  Run tests with coverage report"
	@echo ""
	@echo "Shaders:"
	@echo "  make compile-shaders  Compile all GLSL shaders to SPIR-V"
	@echo "  make verify-shaders   Verify shader compilation"
	@echo ""
	@echo "Building:"
	@echo "  make build          Build wheel and source distribution"
	@echo "  make clean          Remove build artifacts"
	@echo ""
	@echo "Publishing:"
	@echo "  make publish-test   Publish to Test PyPI"
	@echo "  make publish        Publish to production PyPI"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linters (ruff)"
	@echo "  make format         Format code (black, isort)"
	@echo "  make check          Run all quality checks"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) grilly/tests/ -v

test-cpu:
	$(PYTEST) grilly/tests/ -m "not gpu" -v

test-gpu:
	$(PYTEST) grilly/tests/ -k "gpu" -v

test-coverage:
	$(PYTEST) grilly/tests/ --cov=grilly --cov-report=html --cov-report=term -v
	@echo "Coverage report generated in htmlcov/index.html"

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

build: clean
	@echo "Building distribution packages..."
	$(PYTHON) -m build
	@echo "Build complete. Packages in dist/"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."

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
