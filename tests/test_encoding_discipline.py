"""Every repository text read/write must declare its encoding explicitly.

Python's default text encoding is locale-dependent. On a host without
`PYTHONUTF8=1` or a UTF-8 locale, an undeclared `read_text()` can raise
UnicodeDecodeError or silently mis-decode a config, a manifest, or a results
file. The pilot must not depend on an environment variable being set for its
outputs to be readable, so this is enforced statically rather than trusted.

Model-mocked: no weights required. This is a source-level check, in the same
style as `test_source_contains_no_nan_to_zero_collapse`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories that are not ours to police.
SKIP_PARTS = {".venv", ".git", "__pycache__", ".pytest_cache", "results"}

# Modes that are binary, where an encoding argument would be an error.
BINARY_MODE_MARKERS = ("b",)


def _python_files() -> list[Path]:
    return sorted(
        p
        for p in REPO_ROOT.rglob("*.py")
        if not SKIP_PARTS & set(p.relative_to(REPO_ROOT).parts)
    )


def _has_encoding_kwarg(node: ast.Call) -> bool:
    return any(kw.arg == "encoding" for kw in node.keywords)


def _mode_is_binary(node: ast.Call, *, mode_position: int) -> bool:
    """True when the call opens in binary mode, where encoding is not allowed."""
    mode = None
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value
    if mode is None and len(node.args) > mode_position:
        candidate = node.args[mode_position]
        if isinstance(candidate, ast.Constant):
            mode = candidate.value
    if not isinstance(mode, str):
        return False
    return any(marker in mode for marker in BINARY_MODE_MARKERS)


def _label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)  # a temp file, from the self-check below


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)

        if name in {"read_text", "write_text"}:
            if not _has_encoding_kwarg(node):
                found.append(f"{_label(path)}:{node.lineno} {name}()")
        elif name == "open":
            # builtins.open(file, mode, ...) and Path.open(mode, ...)
            mode_position = 1 if name == "open" and not isinstance(func, ast.Attribute) else 0
            if _mode_is_binary(node, mode_position=mode_position):
                continue
            if not _has_encoding_kwarg(node):
                found.append(f"{_label(path)}:{node.lineno} open()")
    return found


def test_no_text_io_without_explicit_encoding():
    """Fails on any new read_text/write_text/open lacking encoding=."""
    violations: list[str] = []
    for path in _python_files():
        violations.extend(_violations(path))
    assert not violations, (
        "text I/O without an explicit encoding (add encoding=\"utf-8\"):\n  "
        + "\n  ".join(violations)
    )


def test_the_check_actually_detects_a_violation(tmp_path):
    """Guard against the guard silently passing because it parses nothing."""
    offender = tmp_path / "bad.py"
    offender.write_text(
        "from pathlib import Path\n"
        "Path('x').read_text()\n"
        "Path('y').write_text('z')\n"
        "open('w')\n",
        encoding="utf-8",
    )
    found = _violations(offender)
    assert len(found) == 3, f"expected 3 violations, found {found}"


def test_binary_modes_are_exempt(tmp_path):
    """encoding= is invalid for binary I/O and must not be demanded."""
    ok = tmp_path / "good.py"
    ok.write_text(
        "from pathlib import Path\n"
        "Path('x').open('rb')\n"
        "open('y', 'wb')\n"
        "open('z', mode='rb')\n"
        "Path('a').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    assert _violations(ok) == []


def test_the_scan_covers_the_real_source_tree():
    """A scan that matched nothing would pass vacuously."""
    files = _python_files()
    names = {p.name for p in files}
    assert len(files) > 20, f"only {len(files)} files scanned"
    for expected in ("run_pilot.py", "decision.py", "internal_only.py", "fetch_etth1.py"):
        assert expected in names, f"{expected} was not scanned"
