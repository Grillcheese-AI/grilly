"""Autodoc helpers to enrich function API pages with practical sections."""

from __future__ import annotations

import inspect
import types
from typing import Any


def _has_heading(lines: list[str], heading: str) -> bool:
    target = heading.strip().lower()
    return any(line.strip().lower() == target for line in lines)


def _annotation_text(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    try:
        return inspect.formatannotation(annotation)
    except Exception:
        return getattr(annotation, "__name__", repr(annotation))


def _safe_repr(value: Any, max_len: int = 60) -> str:
    text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _example_value(param: inspect.Parameter) -> str:
    if param.default is not inspect._empty and isinstance(
        param.default, (bool, int, float, str, type(None))
    ):
        return repr(param.default)

    ann = _annotation_text(param.annotation).lower()
    if "bool" in ann:
        return "False"
    if "int" in ann:
        return "0"
    if "float" in ann:
        return "0.0"
    if "str" in ann:
        return "'example'"
    if "ndarray" in ann or "array" in ann:
        return "np.zeros(1, dtype=np.float32)"
    if "list" in ann or "sequence" in ann:
        return "[]"
    if "dict" in ann or "mapping" in ann:
        return "{}"
    if "tuple" in ann:
        return "()"
    return "None"


def _iter_input_params(signature: inspect.Signature):
    for param in signature.parameters.values():
        if param.name in {"self", "cls"}:
            continue
        yield param


def _dependency_names(obj: Any) -> list[str]:
    module_name = getattr(obj, "__module__", "") or ""
    module_root = module_name.split(".")[0] if module_name else ""
    code = getattr(obj, "__code__", None)
    global_ns = getattr(obj, "__globals__", None)
    if code is None or not isinstance(global_ns, dict):
        return []

    deps: set[str] = set()
    for symbol in code.co_names:
        if symbol not in global_ns:
            continue
        value = global_ns[symbol]
        dep_name = ""
        if isinstance(value, types.ModuleType):
            dep_name = value.__name__
        else:
            dep_name = getattr(value, "__module__", "")
        if not dep_name:
            continue
        dep_root = dep_name.split(".")[0]
        if dep_root in {"builtins", "typing"}:
            continue
        if module_root and dep_root == module_root:
            continue
        deps.add(dep_root)
    return sorted(deps)


def _append_dependencies(lines: list[str], obj: Any) -> None:
    if _has_heading(lines, "Dependencies"):
        return
    dependencies = _dependency_names(obj)
    lines.append("")
    if dependencies:
        dep_text = ", ".join(f"``{dep}``" for dep in dependencies)
        lines.append(f"Dependencies: {dep_text}.")
    else:
        lines.append("Dependencies: ``None`` detected from callable globals.")


def _append_variables(lines: list[str], obj: Any) -> None:
    if _has_heading(lines, "Variables"):
        return

    lines.append("")
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        lines.append("Variables: Signature introspection is unavailable.")
        return

    params = list(_iter_input_params(signature))
    if not params:
        lines.append("Variables: This callable does not take explicit input variables.")
        return

    variable_chunks: list[str] = []
    for param in params:
        annotation = _annotation_text(param.annotation)
        if param.default is inspect._empty:
            variable_chunks.append(f"``{param.name}`` ({annotation}, required)")
        else:
            default_repr = _safe_repr(param.default)
            variable_chunks.append(
                f"``{param.name}`` ({annotation}, optional, default ``{default_repr}``)"
            )

    lines.append(f"Variables: {'; '.join(variable_chunks)}.")


def _append_usage_example(lines: list[str], what: str, name: str, obj: Any) -> None:
    if _has_heading(lines, "Usage Example"):
        return

    module_name = getattr(obj, "__module__", "")
    callable_name = getattr(obj, "__name__", name.split(".")[-1])
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        signature = None

    call_args: list[str] = []
    needs_numpy = False
    if signature is not None:
        for param in _iter_input_params(signature):
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                call_args.append("*args")
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                call_args.append("**kwargs")
                continue

            value = _example_value(param)
            if "np." in value:
                needs_numpy = True

            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                call_args.append(value)
            else:
                call_args.append(f"{param.name}={value}")

    call_expr = ", ".join(call_args)

    lines.extend(["", "Usage Example", "-------------", "", ".. code-block:: python", ""])
    if needs_numpy:
        lines.append("   import numpy as np")

    if what == "method":
        class_name = getattr(obj, "__qualname__", "Class.method").split(".")[0]
        if module_name:
            lines.append(f"   from {module_name} import {class_name}")
            lines.append("")
        lines.append(f"   instance = {class_name}(...)")
        lines.append(f"   result = instance.{callable_name}({call_expr})")
    else:
        if module_name:
            lines.append(f"   from {module_name} import {callable_name}")
            lines.append("")
        lines.append(f"   result = {callable_name}({call_expr})")

    lines.append("")


def _process_docstring(app, what: str, name: str, obj: Any, options, lines: list[str]) -> None:
    if what not in {"function", "method"}:
        return

    if not any(line.strip() for line in lines):
        lines.append(f"Autogenerated reference for ``{name}``.")

    _append_dependencies(lines, obj)
    _append_variables(lines, obj)
    _append_usage_example(lines, what, name, obj)


def setup(app):
    app.connect("autodoc-process-docstring", _process_docstring)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
