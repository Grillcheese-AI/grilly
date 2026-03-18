# Contributing

Grilly is open source under the MIT license. Contributions are welcome.

---

## Getting Started

1. Fork the repository
2. Clone and install in development mode:

```bash
git clone https://github.com/YOUR_USERNAME/grilly.git
cd grilly
pip install -e ".[dev]"
```

3. Build the C++ backend (see [Installation](getting-started/installation.md))

---

## Development Workflow

1. Create a feature branch from `main`
2. Write code and tests
3. Run linting and tests
4. Submit a pull request

---

## Code Style

- **Line length**: 100 characters
- **Formatter**: Black (`black . --check`)
- **Linter**: Ruff (`ruff check .`)
- **Import sorting**: isort with Black profile (`isort . --check-only`)
- **Type checking**: mypy (`mypy .`)
- **Target Python**: 3.12

```bash
# Run all checks
ruff check .
black . --check
isort . --check-only
mypy .
```

---

## Testing

```bash
# All tests (requires Vulkan)
uv run pytest tests/ -v

# CPU-only tests (no GPU needed)
uv run pytest tests/ -m "not gpu" -v

# With coverage
uv run pytest tests/ --cov=. --cov-report=term

# Single test
pytest tests/test_snn.py -k "test_lif"
```

GPU tests auto-skip when Vulkan is unavailable. The `gpu_backend` fixture in `tests/conftest.py` handles this.

---

## Adding a New Layer

1. Create the GLSL shader in `shaders/` (forward + backward)
2. Compile to SPIR-V: `glslc shader.glsl -o spv/shader.spv`
3. Add the backend dispatch in `backend/` (bridge function or VulkanCompute method)
4. Create the `nn.Module` subclass in `nn/`
5. Add functional API wrapper in `functional/`
6. Write tests in `tests/`
7. Export from `nn/__init__.py` and `functional/__init__.py`

---

## File Organization Rules

- Maximum 1000 lines per file
- Proper OOP -- no God classes
- Always edit existing files rather than creating duplicates
- Each `nn.Module` in its own file (or grouped by category)

---

## Pull Request Checklist

- [ ] Tests pass: `uv run pytest tests/ -v`
- [ ] Linting passes: `ruff check .`
- [ ] New features have tests
- [ ] New public API is exported from `__init__.py`
- [ ] Shaders compiled to SPIR-V and included

---

## Reporting Issues

File issues at [github.com/grillcheese-ai/grilly/issues](https://github.com/grillcheese-ai/grilly/issues). Include:

- Grilly version (`python -c "import grilly; print(grilly.__version__)"`)
- GPU model and driver version (`vulkaninfo --summary`)
- OS and Python version
- Minimal reproduction steps
