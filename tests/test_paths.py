# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Portability guards: nothing may depend on where the checkout lives.

A hardcoded absolute path is the kind of defect that never fails for the person
who wrote it. `run_n50_gemini_vs_claude_sweetspot.py` carried one contributor's
`/usr/local/google/home/.../straitjacket/src` for months and was simply
unrunnable anywhere else. These tests pin the two halves of the fix:

  * source files may not contain absolute path literals -- environment-specific
    locations come from env vars or are derived from `__file__`;
  * paths that reach a human or a persisted artifact are rendered relative to
    the repo, so a console log or a committed quarantine file does not carry
    somebody's home directory.
"""

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import paths  # noqa: E402


# ==============================================================================
# --- RENDERING ---
# ==============================================================================

def test_display_makes_inside_paths_relative():
    inside = os.path.join(ROOT, "src", "datasets.py")
    assert paths.display(inside) == os.path.join("src", "datasets.py")


def test_display_leaves_outside_paths_alone():
    """A file genuinely outside the checkout reads better in full than as a
    pile of `../`."""
    outside = os.path.join(os.sep, "somewhere", "else", "file.txt")
    assert paths.display(outside) == outside


def test_display_passes_through_empty():
    assert paths.display("") == ""
    assert paths.display(None) is None


def test_scrub_replaces_repo_root():
    text = f"File \"{os.path.join(ROOT, 'x', 'prog.py')}\", line 3"
    out = paths.scrub(text)
    assert ROOT not in out
    assert "<repo>" in out


def test_scrub_replaces_home():
    home = os.path.expanduser("~")
    if home == os.sep:
        pytest.skip("no meaningful home directory here")
    out = paths.scrub(f"cache at {os.path.join(home, '.cache', 'sj', 'prog.py')}")
    assert home not in out
    assert out.startswith("cache at ~")


def test_scrub_prefers_repo_over_home():
    """The repo may sit inside the home directory; substituting home first
    would mangle a path that should have read `<repo>/...`."""
    inside = os.path.join(ROOT, "classeval", "prog.py")
    out = paths.scrub(f"at {inside}")
    assert "<repo>" in out


# ==============================================================================
# --- NO ABSOLUTE PATHS IN SOURCE ---
# ==============================================================================

# Matches a quoted POSIX absolute path with at least two segments, which is what
# a machine-specific location looks like. Single-segment roots ("/", "/tmp")
# and the sandbox-internal paths the harness constructs are not the target.
ABS_LITERAL = re.compile(r"""['"](/(?:usr|home|Users|opt|private|var|mnt|media)/[^'"\n]{4,})['"]""")

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "data", "results",
             "archive", "reports"}


def _python_sources():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def test_no_absolute_path_literals_in_python_sources():
    offenders = []
    for path in _python_sources():
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue          # prose, including this module's own docs
                m = ABS_LITERAL.search(line)
                if m:
                    offenders.append(f"{paths.display(path)}:{lineno}: {m.group(1)}")
    assert not offenders, (
        "absolute path literals make these files environment-specific; take the "
        "location from an env var or derive it from __file__:\n  "
        + "\n  ".join(offenders))


def test_sj_src_is_an_env_var_everywhere():
    """The escape hatch for a source checkout of the harness is SJ_SRC, and it
    must be read from the environment, never baked in."""
    for path in _python_sources():
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        for m in re.finditer(r"^SJ_SRC\s*=\s*(.+)$", text, re.M):
            assert "os.environ" in m.group(1), (
                f"{paths.display(path)} hardcodes SJ_SRC: {m.group(1).strip()}")
