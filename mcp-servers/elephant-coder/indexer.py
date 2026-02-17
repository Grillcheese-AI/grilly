"""
AST-based code indexer for elephant-coder.

Compresses Python source files into MemoryEntry capsules using ast parsing.
This is the "Capsule Encoding" step — analogous to CapsuleProject (nn/capsule.py)
which compresses 384D embeddings into 32D capsules.

The keyword extraction is the "Dentate Gyrus" step — producing a sparse set of
discriminative tokens for efficient FTS5 retrieval (pattern separation).

Also supports basic TS/JS indexing via regex extraction.
"""

import ast
import logging
import os
import re
from pathlib import Path

from memory_store import MemoryEntry, make_memory_id

logger = logging.getLogger("elephant-coder.indexer")

# Max chars for docstring excerpts
_MAX_DOC = 200
# Max methods to list in a class summary
_MAX_METHODS = 40


def index_file(file_path: str) -> list[MemoryEntry]:
    """Parse a source file and produce MemoryEntry objects for each symbol.

    Supports Python (.py) via AST and TypeScript/JavaScript (.ts/.js) via regex.
    """
    path = Path(file_path)
    if not path.exists():
        return []

    if path.suffix == ".py":
        return _index_py_file(file_path)
    elif path.suffix in (".ts", ".js", ".tsx", ".jsx"):
        return _index_ts_file(file_path)
    return []


def _index_py_file(file_path: str) -> list[MemoryEntry]:
    """Parse a Python file and produce MemoryEntry objects for each symbol."""
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError) as exc:
        logger.debug("Failed to parse %s: %s", file_path, exc)
        return []

    file_mtime = os.path.getmtime(file_path)
    entries: list[MemoryEntry] = []

    # Module-level entry
    entries.append(_module_entry(file_path, tree, source, file_mtime))

    # Top-level classes and functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            entries.append(_class_entry(file_path, node, source, file_mtime))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entries.append(_function_entry(file_path, node, source, file_mtime))

    return entries


# ------------------------------------------------------------------
# TypeScript/JavaScript indexing via regex
# ------------------------------------------------------------------

_TS_FUNCTION_RE = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
    re.MULTILINE,
)
_TS_ARROW_RE = re.compile(
    r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*\w+)?\s*=>",
    re.MULTILINE,
)
_TS_CLASS_RE = re.compile(
    r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?\s*\{",
    re.MULTILINE,
)
_TS_EXPORT_RE = re.compile(
    r"export\s+(?:default\s+)?(?:const|let|var|type|interface|enum)\s+(\w+)",
    re.MULTILINE,
)


def _index_ts_file(file_path: str) -> list[MemoryEntry]:
    """Index a TypeScript/JavaScript file via regex extraction."""
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    file_mtime = os.path.getmtime(file_path)
    entries: list[MemoryEntry] = []
    module_name = path.stem

    functions = []
    classes = []
    exports = []

    # Extract functions
    for m in _TS_FUNCTION_RE.finditer(source):
        name = m.group(1)
        params = m.group(2).strip()
        functions.append(name)
        entries.append(MemoryEntry(
            memory_id=make_memory_id(file_path, name, "function"),
            file_path=file_path,
            symbol_name=name,
            kind="function",
            summary=f"function {name}({params})",
            keywords=_extract_keywords_from_strings([name, params]),
            file_mtime=file_mtime,
        ))

    # Extract arrow functions
    for m in _TS_ARROW_RE.finditer(source):
        name = m.group(1)
        if name not in functions:
            functions.append(name)
            entries.append(MemoryEntry(
                memory_id=make_memory_id(file_path, name, "function"),
                file_path=file_path,
                symbol_name=name,
                kind="function",
                summary=f"const {name} = (...) =>",
                keywords=_extract_keywords_from_strings([name]),
                file_mtime=file_mtime,
            ))

    # Extract classes
    for m in _TS_CLASS_RE.finditer(source):
        name = m.group(1)
        extends = m.group(2) or ""
        classes.append(name)
        summary = f"class {name}"
        if extends:
            summary += f" extends {extends}"
        deps = [extends] if extends else []
        entries.append(MemoryEntry(
            memory_id=make_memory_id(file_path, name, "class"),
            file_path=file_path,
            symbol_name=name,
            kind="class",
            summary=summary,
            keywords=_extract_keywords_from_strings([name, extends]),
            dependencies=deps,
            file_mtime=file_mtime,
        ))

    # Extract other exports
    for m in _TS_EXPORT_RE.finditer(source):
        name = m.group(1)
        if name not in functions and name not in classes:
            exports.append(name)

    # Module entry
    top_symbols = [f"function {f}" for f in functions] + [f"class {c}" for c in classes] + exports
    summary_parts = []
    if top_symbols:
        summary_parts.append("Defines: " + ", ".join(top_symbols))

    entries.insert(0, MemoryEntry(
        memory_id=make_memory_id(file_path, module_name, "module"),
        file_path=file_path,
        symbol_name=module_name,
        kind="module",
        summary="\n".join(summary_parts) if summary_parts else f"Module {module_name}",
        keywords=_extract_keywords_from_strings([module_name] + functions + classes + exports),
        file_mtime=file_mtime,
    ))

    return entries


# ------------------------------------------------------------------
# Python entry builders
# ------------------------------------------------------------------


def _module_entry(file_path: str, tree: ast.Module, source: str, file_mtime: float) -> MemoryEntry:
    """Build a module-level summary."""
    doc = _get_docstring(tree)
    imports = _extract_imports(tree)
    all_names = _extract_all(tree)
    constants = _extract_constants(tree)
    top_symbols = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            top_symbols.append(f"class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_symbols.append(f"def {node.name}")

    summary_parts = []
    if doc:
        summary_parts.append(doc[:_MAX_DOC])
    if all_names:
        summary_parts.append(f"__all__ = {all_names}")
    if top_symbols:
        summary_parts.append("Defines: " + ", ".join(top_symbols))
    if constants:
        summary_parts.append("Constants: " + ", ".join(constants[:10]))

    module_name = Path(file_path).stem
    keywords = _extract_keywords_from_strings(
        [module_name, doc or ""] + top_symbols + imports + constants
    )

    return MemoryEntry(
        memory_id=make_memory_id(file_path, module_name, "module"),
        file_path=file_path,
        symbol_name=module_name,
        kind="module",
        summary="\n".join(summary_parts) if summary_parts else f"Module {module_name}",
        keywords=keywords,
        dependencies=imports,
        file_mtime=file_mtime,
    )


def _class_entry(file_path: str, node: ast.ClassDef, source: str, file_mtime: float) -> MemoryEntry:
    """Build a class summary: name, bases, inheritance chain, docstring, method signatures."""
    doc = _get_docstring(node)
    bases = [_unparse_safe(b) for b in node.bases]

    # Extract full inheritance chain info from bases
    inheritance = bases[:]

    methods = []
    attrs = []
    properties = []
    class_vars = []
    decorators = [_unparse_safe(d) for d in node.decorator_list]

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _function_signature(item)
            # Check for property decorators
            for d in item.decorator_list:
                dname = _unparse_safe(d)
                if "property" in dname:
                    properties.append(item.name)
                    break
            methods.append(sig)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            ann = _unparse_safe(item.annotation) if item.annotation else "?"
            attrs.append(f"{item.target.id}: {ann}")
        elif isinstance(item, ast.Assign):
            # Class variables
            for target in item.targets:
                if isinstance(target, ast.Name):
                    class_vars.append(target.id)

    summary_parts = []
    head = f"class {node.name}"
    if bases:
        head += f"({', '.join(bases)})"
    if decorators:
        for dec in decorators:
            summary_parts.append(f"@{dec}")
    summary_parts.append(head)
    if doc:
        summary_parts.append(doc[:_MAX_DOC])
    if attrs:
        summary_parts.append("Attrs: " + ", ".join(attrs[:10]))
    if class_vars:
        summary_parts.append("ClassVars: " + ", ".join(class_vars[:10]))
    if properties:
        summary_parts.append("Properties: " + ", ".join(properties[:10]))
    if methods:
        shown = methods[:_MAX_METHODS]
        summary_parts.append("Methods:\n  " + "\n  ".join(shown))
        if len(methods) > _MAX_METHODS:
            summary_parts.append(f"  ... +{len(methods) - _MAX_METHODS} more")

    # Richer keywords: include method names, parameter names, decorator names, type annotations
    method_names = [m.split("(")[0].replace("def ", "").replace("async def ", "") for m in methods]
    keywords = _extract_keywords_from_strings(
        [node.name, doc or ""]
        + inheritance
        + method_names
        + decorators
        + class_vars
        + [a.split(":")[0].strip() for a in attrs]
    )
    deps = bases + _extract_calls(node)

    return MemoryEntry(
        memory_id=make_memory_id(file_path, node.name, "class"),
        file_path=file_path,
        symbol_name=node.name,
        kind="class",
        summary="\n".join(summary_parts),
        keywords=keywords,
        dependencies=deps,
        file_mtime=file_mtime,
    )


def _function_entry(
    file_path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    file_mtime: float,
) -> MemoryEntry:
    """Build a function summary: signature, docstring, return type, calls."""
    doc = _get_docstring(node)
    sig = _function_signature(node)
    ret = _unparse_safe(node.returns) if node.returns else None
    calls = _extract_calls(node)
    decorators = [_unparse_safe(d) for d in node.decorator_list]

    # Extract parameter names and type annotations for keywords
    param_names = [arg.arg for arg in node.args.args if arg.arg != "self"]
    param_types = []
    for arg in node.args.args:
        if arg.annotation:
            param_types.append(_unparse_safe(arg.annotation))

    summary_parts = []
    if decorators:
        for dec in decorators:
            summary_parts.append(f"@{dec}")
    summary_parts.append(sig)
    if doc:
        summary_parts.append(doc[:_MAX_DOC])
    if ret:
        summary_parts.append(f"Returns: {ret}")
    if calls:
        summary_parts.append(f"Calls: {', '.join(calls[:15])}")

    keywords = _extract_keywords_from_strings(
        [node.name, doc or ""]
        + calls
        + param_names
        + param_types
        + decorators
        + ([ret] if ret else [])
    )

    return MemoryEntry(
        memory_id=make_memory_id(file_path, node.name, "function"),
        file_path=file_path,
        symbol_name=node.name,
        kind="function",
        summary="\n".join(summary_parts),
        keywords=keywords,
        dependencies=calls,
        file_mtime=file_mtime,
    )


# ------------------------------------------------------------------
# AST helpers
# ------------------------------------------------------------------


def _get_docstring(node: ast.AST) -> str | None:
    """Extract docstring from a module, class, or function node."""
    try:
        doc = ast.get_docstring(node)
        return doc.strip() if doc else None
    except Exception:
        return None


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Produce a compact signature string like 'def foo(x: int, y: str) -> bool'."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    params = []
    for arg in node.args.args:
        ann = f": {_unparse_safe(arg.annotation)}" if arg.annotation else ""
        params.append(f"{arg.arg}{ann}")
    if node.args.vararg:
        params.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        params.append(f"**{node.args.kwarg.arg}")

    ret = f" -> {_unparse_safe(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({', '.join(params)}){ret}"


def _extract_imports(tree: ast.Module) -> list[str]:
    """Extract import names from a module AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    return sorted(set(imports))


def _extract_all(tree: ast.Module) -> list[str] | None:
    """Extract __all__ list if defined."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return [_unparse_safe(elt) for elt in node.value.elts]
    return None


def _extract_constants(tree: ast.Module) -> list[str]:
    """Extract module-level constant assignments (UPPER_CASE names)."""
    constants = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() and len(target.id) >= 2:
                    val = _unparse_safe(node.value)
                    if len(val) < 60:
                        constants.append(f"{target.id} = {val}")
                    else:
                        constants.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper() and len(node.target.id) >= 2:
                constants.append(node.target.id)
    return constants


def _extract_calls(node: ast.AST) -> list[str]:
    """Extract function/method call names from an AST subtree."""
    calls = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.add(child.func.attr)
    return sorted(calls)


def _unparse_safe(node: ast.AST | None) -> str:
    """Safely unparse an AST node to string."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def _extract_keywords_from_strings(strings: list[str]) -> list[str]:
    """Extract discriminative keywords from a list of strings.

    Splits camelCase and snake_case identifiers, lowercases, deduplicates.
    This is the "DG sparse expansion" — producing a sparse set of tokens
    that disambiguate this entry from others.
    """
    tokens: set[str] = set()
    for s in strings:
        if not s:
            continue
        # Split on non-alphanumeric
        parts = re.split(r"[^a-zA-Z0-9]+", s)
        for part in parts:
            if not part or len(part) < 2:
                continue
            # Split camelCase
            sub = re.sub(r"([a-z])([A-Z])", r"\1 \2", part).split()
            for token in sub:
                low = token.lower()
                if len(low) >= 2 and low not in _STOP_WORDS:
                    tokens.add(low)
    return sorted(tokens)


# Common words that don't help discriminate code symbols
_STOP_WORDS = frozenset(
    {
        "the",
        "is",
        "in",
        "of",
        "to",
        "and",
        "or",
        "for",
        "if",
        "not",
        "with",
        "as",
        "from",
        "by",
        "on",
        "at",
        "be",
        "it",
        "an",
        "no",
        "do",
        "self",
        "cls",
        "none",
        "true",
        "false",
        "return",
        "def",
        "class",
        "import",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "any",
        "optional",
    }
)
