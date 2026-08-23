# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Repo-relative path rendering, so output does not depend on where the checkout is.

Every path printed by a runner or written into a results file used to be
absolute, which meant a console log or a committed artifact carried the
author's home directory. That is noise in a shared repository and, worse, it is
misleading: a quarantine file whose evidence names
`/usr/local/google/home/<user>/.../prog.py` reads as if that path mattered to
the finding, when the only thing that mattered was the failure.

`display()` is for anything a human reads; `scrub()` is for anything persisted.
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPO_ROOT)


def display(path, root=None):
    """Render `path` relative to the repository root when it lives inside it.

    Falls back to the path as given -- a file genuinely outside the checkout is
    better shown in full than as a pile of `../`.
    """
    if not path:
        return path
    base = os.path.abspath(root or REPO_ROOT)
    full = os.path.abspath(str(path))
    try:
        rel = os.path.relpath(full, base)
    except ValueError:            # different drive on Windows
        return str(path)
    return str(path) if rel.startswith(os.pardir) else rel


def scrub(text, root=None):
    """Replace machine-specific prefixes in captured output before persisting.

    The repository root becomes `<repo>` and the user's home becomes `~`, in
    that order, so a path inside the checkout is not first mangled by the home
    substitution.
    """
    if not text:
        return text
    out = str(text)
    base = os.path.abspath(root or REPO_ROOT)
    out = out.replace(base, "<repo>")
    home = os.path.expanduser("~")
    if home and home != os.sep:
        out = out.replace(home, "~")
    return out
