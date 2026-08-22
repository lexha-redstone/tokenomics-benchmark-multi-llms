# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Real straitjacket (``ctx-harness``) integration for the multi-LLM benchmark.

This module is the ONLY place the benchmark is allowed to produce a
"Straitjacket digest". It never re-implements the harness's evidence
selection: profile detection, digest rendering, coverage receipts, span
minting and bounded retrieval all come from the upstream ``ctx`` package
exactly as published.

Why this exists
---------------
The previous integration triaged a *string that had already been truncated*
by filtering it for keywords (``"FAIL:"``, ``"AssertionError"``, ...). Both
halves of that are anti-patterns the upstream project names explicitly:

* ``head``/``tail`` truncation "is cheap because someone else pays" — the
  evidence is destroyed before anything gets to select from it;
* keyword/position filtering has no coverage receipt and no retrieval
  address, so omission becomes amnesia.

straitjacket's invariant is a *birth-time* gate:

    Potentially unbounded output must be captured before it reaches the
    model or rejected before execution.

So the benchmark now executes candidate solutions **through** the harness
(``ctx.execution.run_capture``), renders the digest with the upstream
profile registry (``ctx.digest.render_run_digest``), and keeps the complete
output addressable in the artifact store (``ctx.retrieval.get`` / ``search``
/ ``ctx.rundiff.run_diff``).

Backends
--------
``library``  in-process ``import ctx`` — preferred, exact, gives the raw
             stream bytes back for the *uncontained* baseline arms.
``cli``      shells out to the published ``ctx`` executable. Same digest
             (same code), but raw-stream recovery goes back through bounded
             retrieval, so ``raw_exact`` is False.
``off``      harness genuinely unavailable. Nothing fabricates a digest;
             callers must degrade loudly (see :func:`require`).

Environment
-----------
``SJ_BACKEND``      ``auto`` (default) | ``library`` | ``cli`` | ``off``
``SJ_HOME``         harness home (default ``~/.cache/tokenomics-sj``; kept SHORT
                    and out of the repo because the sandbox path is printed in
                    every traceback and competes for the digest's evidence budget)
``SJ_WORKSPACE``    workspace root (default ``<SJ_HOME>/ws``)
``CTX_STATE_HOME``  artifact store root (default ``<SJ_HOME>/state``)
``SJ_RAW_CAP``      chars of raw log handed to *uncontained* arms — the single
                    native-truncation knob (default 2500; ``0`` = the true,
                    uncapped flood)
``SJ_KEEP_SANDBOX`` ``1`` keeps every per-task sandbox for inspection (debug
                    mode: unique paths make digests non-reproducible again)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "SJUnavailable",
    "ContainedRun",
    "status",
    "available",
    "require",
    "contained_run",
    "contain_text",
    "estimate_tokens",
    "raw_cap",
    "tail_to_cap",
    "DETERMINISTIC_UNITTEST_TAIL",
    "CAPTURE_ENV",
    "new_sandbox",
    "drop_sandbox",
    "workspace_root",
    "frame_budget",
    "FRAME_LINE_BUDGET",
]

# The header is normally the first line, but the CLI may print a reflex
# notice above it ("densified: re-run detected …"), so it is located rather
# than assumed.
_DIGEST_HEADER_RE = re.compile(r"^\[ctx run:([0-9a-f]+) profile=([^\]]+)\]", re.MULTILINE)


class SJUnavailable(RuntimeError):
    """The straitjacket harness is not installed / not usable."""


# ==============================================================================
# --- ENVIRONMENT & BACKEND RESOLUTION ---
# ==============================================================================

def _sj_home() -> Path:
    """Root for the harness workspace and artifact store.

    Deliberately NOT inside the repository. The sandbox path is not private
    bookkeeping — it is printed inside every Python traceback the tests
    produce, so it lands in the captured bytes and competes for the digest's
    per-line evidence budget (:data:`FRAME_LINE_BUDGET`). This checkout's own
    path is 83 characters before anything is appended to it, which was enough
    to push ``, line 6, in test_a`` off the end of the innermost-frame row —
    the line number is the whole reason that row exists.

    A short, repo-independent cache directory keeps the frame intact and the
    digest identical from machine to machine. Override with ``SJ_HOME``.
    """
    env = os.environ.get("SJ_HOME")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "tokenomics-sj"


def workspace_root() -> Path:
    root = Path(os.environ.get("SJ_WORKSPACE") or (_sj_home() / "ws")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _state_home() -> Path:
    """Artifact-store root. Kept in a benchmark-owned directory so a sweep
    never pollutes the user's real ``ctx`` store."""
    env = os.environ.get("CTX_STATE_HOME")
    if env:
        return Path(env)
    root = _sj_home() / "state"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["CTX_STATE_HOME"] = str(root)
    return root


# The uncontained path's own tail truncation. 2,500 chars is what this
# repository's arms have always effectively sent (`err[-2500:]`), so the
# baseline stays comparable with previously published runs.
#
# ONE cap, in one place. It used to be two — 4,000 here and 2,500 again inside
# the arms — which meant the measured "native baseline" was 4,000 chars while
# the native arm actually sent 2,500, and the arm scored a 55% improvement
# over itself.
DEFAULT_RAW_CAP = 2500

# Bytes native_stream may peek at to decide which stream carried the failure.
# Bounded so the decision itself cannot pull a flood into memory.
_NATIVE_PROBE = 4096


def raw_cap() -> int:
    """Chars of raw log the *uncontained* arms send.

    This is the native path's own truncation, not straitjacket's: it exists so
    a single 300k-token flood cannot bankrupt a sweep. Set ``SJ_RAW_CAP=0`` to
    measure the real, uncapped native cost. Whatever the cap, the full size is
    always recorded on the :class:`ContainedRun`, so a report can show what the
    native path would have paid.
    """
    try:
        return max(0, int(os.environ.get("SJ_RAW_CAP", str(DEFAULT_RAW_CAP))))
    except ValueError:
        return DEFAULT_RAW_CAP


def tail_to_cap(text: str) -> str:
    """Apply the uncontained path's tail truncation. Idempotent.

    The single implementation of that cut. Applying it to an already-capped
    payload is a no-op, so callers can enforce the invariant defensively
    without introducing a second budget.
    """
    cap = raw_cap()
    text = str(text)
    return text[-cap:] if cap and len(text) > cap else text


# ==============================================================================
# --- DETERMINISTIC CAPTURE ---
# ==============================================================================
#
# Artifact identity is the sha256 of the captured bytes, so anything a runner
# prints that varies between otherwise identical executions mints a new blob,
# a new manifest and a new run handle for the same failure.
#
# ``unittest.TextTestRunner`` prints ``Ran 12 tests in 0.114s``. The elapsed
# time is not evidence. Measured across four separate invocations of one
# identical failing program: 0.114s / 0.111s / 0.111s / 0.112s — three distinct
# handles for one failure, even though the rendered digest body was
# byte-identical every time (the profile never surfaces the timing line).
#
# Upstream is behaving correctly: it guarantees a stable digest *for stable
# captured bytes*. Removing the jitter is this harness's job, and it is our own
# runner code — not straitjacket's — that emits it. The line is kept, because
# the unittest profile detects on ``Ran N tests in ``; only its digits are
# pinned.
DETERMINISTIC_UNITTEST_TAIL = '''

import re as _re, sys as _sys, unittest as _ut
_ut.TestCase.maxDiff = None


class _SteadyStream:
    """stderr proxy that pins the runner's elapsed-time digits."""

    _RAN = _re.compile(r"(Ran \\d+ tests? in )[\\d.]+s")

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        self._stream.write(self._RAN.sub(r"\\g<1>0.000s", text))

    def flush(self):
        self._stream.flush()


_res = _ut.TextTestRunner(verbosity=0, stream=_SteadyStream(_sys.stderr)).run(
    _ut.TestLoader().loadTestsFromTestCase(TestCases))
_sys.exit(0 if _res.wasSuccessful() else 1)
'''

# Environment every captured child inherits. Both entries exist to keep the
# child from writing run-to-run noise into the artifact: a GUI backend fails or
# warns differently per host, and hash randomisation reorders any set a
# candidate solution prints.
CAPTURE_ENV = {"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}


def estimate_tokens(n_bytes: int) -> int:
    """Upstream's deterministic estimator, so our numbers match ``ctx``'s."""
    try:
        from ctx.textutil import estimate_tokens as _est
        return _est(n_bytes)
    except Exception:
        return max(1, n_bytes // 4) if n_bytes else 0


_state: dict = {"resolved": False, "backend": "off", "version": None, "reason": ""}
_lock = threading.Lock()
_ws = None
_store = None


def _resolve() -> dict:
    global _ws, _store
    with _lock:
        if _state["resolved"]:
            return _state
        _state["resolved"] = True
        want = (os.environ.get("SJ_BACKEND") or "auto").strip().lower()

        if want == "off":
            _state.update(backend="off", reason="disabled via SJ_BACKEND=off")
            return _state

        if want in ("auto", "library"):
            _state_home()
            try:
                import ctx  # noqa: F401
                from ctx.store import Store
                from ctx.workspace import resolve_workspace

                _ws = resolve_workspace(str(workspace_root()))
                _store = Store(_ws.workspace_id,
                               retention_days=_ws.config.store.retention_days)
                _state.update(backend="library", version=getattr(ctx, "__version__", "?"))
                return _state
            except Exception as e:  # pragma: no cover - environment dependent
                _state["reason"] = f"library import failed: {type(e).__name__}: {e}"
                if want == "library":
                    return _state

        if want in ("auto", "cli"):
            _state_home()
            exe = shutil.which("ctx")
            if exe:
                try:
                    out = subprocess.run([exe, "--version"], capture_output=True,
                                         text=True, timeout=30)
                    ver = (out.stdout or out.stderr).strip().split()[-1] if out.returncode == 0 else "?"
                except Exception:
                    ver = "?"
                _state.update(backend="cli", version=ver, reason="")
                return _state
            if want == "cli":
                _state["reason"] = "`ctx` not found on PATH"
                return _state
            _state["reason"] = (_state["reason"] + "; " if _state["reason"] else "") + \
                               "`ctx` not found on PATH"

        _state["backend"] = "off"
        return _state


def status() -> dict:
    """Machine-readable harness status — record this in every result file."""
    st = _resolve()
    out = {
        "backend": st["backend"],
        "available": st["backend"] != "off",
        "ctx_version": st["version"],
        "reason": st["reason"],
        "workspace": str(workspace_root()) if st["backend"] != "off" else None,
        "state_home": os.environ.get("CTX_STATE_HOME"),
        "raw_cap_chars": raw_cap(),
    }
    if st["backend"] != "off":
        fb = frame_budget()
        out["frame_budget"] = fb
        if not fb["frame_fits"]:
            _warn_once(
                "straitjacket: sandbox path is "
                f"{fb['frame_chars_used']} chars against a {fb['frame_chars_budget']}-char "
                "evidence budget — traceback line numbers will be clipped out of "
                "every digest. Set SJ_HOME or SJ_WORKSPACE to a shorter path."
            )
    return out


_warned: set = set()


def _warn_once(message: str) -> None:
    if message in _warned:
        return
    _warned.add(message)
    print(message, file=sys.stderr)


def available() -> bool:
    return _resolve()["backend"] != "off"


INSTALL_HINT = (
    "straitjacket harness not available. Install it before running any "
    "straitjacket arm:\n"
    "    python -m pip install --upgrade ctx-harness\n"
    "(or `pip install -e /path/to/straitjacket`). Set SJ_BACKEND=off only if "
    "you intend to run the uncontained arms alone."
)


def require() -> None:
    """Refuse to proceed rather than silently substituting a fake digest.

    A benchmark row labelled "Straitjacket UnittestProfile ($0.00)" must be
    produced by the real profile. If it cannot be, the run has to stop.
    """
    st = _resolve()
    if st["backend"] == "off":
        raise SJUnavailable(f"{INSTALL_HINT}\nreason: {st['reason'] or 'unknown'}")


# ==============================================================================
# --- CONTAINED RUN ---
# ==============================================================================

@dataclass
class ContainedRun:
    """One captured execution: a bounded digest plus an addressable artifact.

    ``digest`` is the model-visible payload (what a straitjacket arm sends);
    :meth:`native_payload` is what an uncontained arm sends. ``raw_exact``
    says whether the stored bytes come back verbatim (library) or through
    bounded retrieval (cli).
    """

    handle: str                      # "run:080a40080b4b"
    short_id: str                    # "080a40080b4b"
    exit_code: int | None
    timed_out: bool
    profile: str                     # "unittest/v1"
    digest: str
    raw_exact: bool = True
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_lines: int = 0
    stderr_lines: int = 0
    stdout_blob: str = ""
    stderr_blob: str = ""
    backend: str = "library"
    # CLI backend only: streams recovered through bounded retrieval rather
    # than read from the store. Empty on the library path.
    _recovered: dict = field(default_factory=dict, repr=False)
    _manifest: dict = field(default_factory=dict, repr=False)

    # -- payloads -----------------------------------------------------------
    #
    # Read from the store on demand, and by default only the tail. run_capture
    # streams the child's output straight to disk so a flood never sits in
    # memory; materialising both streams eagerly here would undo exactly that.
    # Measured: a 40.9 MB stdout capture pushed the harness process to a
    # 150.8 MB peak heap to produce a 663-char digest and a 506-char native
    # payload.

    def raw_tail(self, stream: str = "stderr", nbytes: int | None = None) -> str:
        """Last ``nbytes`` of a captured stream (all of it when ``None``).

        A tail may start mid-codepoint; decoding is lossy-tolerant, which
        costs at most one replacement character at the cut.
        """
        if stream in self._recovered:
            text = self._recovered[stream]
            return text[-nbytes:] if nbytes and len(text) > nbytes else text
        blob = self.stdout_blob if stream == "stdout" else self.stderr_blob
        size = self.stdout_bytes if stream == "stdout" else self.stderr_bytes
        if not blob or not size:
            return ""
        return _read_blob(blob, size, nbytes)

    @property
    def raw_stdout(self) -> str:
        """The complete stdout. Unbounded — for tests and debugging, never on
        the per-task path."""
        return self.raw_tail("stdout")

    @property
    def raw_stderr(self) -> str:
        """The complete stderr. Unbounded — see :attr:`raw_stdout`."""
        return self.raw_tail("stderr")

    @property
    def native_stream(self) -> str:
        """Which stream an untreated harness would have read.

        stderr when it has anything: that is where unittest/pytest put the
        failure, and it is what this repository's pre-containment baseline
        read (``r.stderr.strip()[-4000:]``). stdout only as the fallback, for
        runners that report there.
        """
        if self.stderr_bytes and self.raw_tail("stderr", _NATIVE_PROBE).strip():
            return "stderr"
        return "stdout"

    def native_payload(self) -> str:
        """What an *uncontained* arm sends to the model.

        This has to be the baseline as the benchmark actually defined it, not
        a worse one. Concatenating stdout+stderr and tail-truncating the pair
        spends the whole budget on stdout chatter the untreated path never
        forwarded — which inflates the native arm's token cost and degrades
        the evidence it repairs from, biasing every comparison toward
        straitjacket. The uncontained arm gets the same stream it always got,
        tail-truncated the same way (see :func:`raw_cap`).
        """
        cap = raw_cap()
        # A little slack so stripping trailing whitespace cannot leave the
        # payload short of the cap.
        raw = self.raw_tail(self.native_stream, (cap + 64) if cap else None)
        return tail_to_cap(raw.strip())

    # -- metrics ------------------------------------------------------------
    @property
    def raw_bytes(self) -> int:
        return self.stdout_bytes + self.stderr_bytes

    def metrics(self) -> dict:
        """Three different numbers that are easy to conflate:

        ``raw_tokens_est``            everything the execution produced.
        ``native_sent_tokens_est``    what the untreated baseline forwards.
        ``digest_tokens_est``         what the contained arm forwards.

        ``raw - digest`` is containment (captured, attested, addressable, but
        never resident). ``native - digest`` is the A/B advantage. They are
        not the same number, and reporting the first as if it were the second
        overstates the mechanism.
        """
        raw_tok = estimate_tokens(self.raw_bytes)
        dig_tok = estimate_tokens(len(self.digest.encode("utf-8")))
        nat_tok = estimate_tokens(len(self.native_payload().encode("utf-8")))
        return {
            "sj_handle": self.handle,
            "sj_profile": self.profile,
            "sj_backend": self.backend,
            "raw_bytes": self.raw_bytes,
            "raw_lines": self.stdout_lines + self.stderr_lines,
            "raw_tokens_est": raw_tok,
            "native_stream": self.native_stream,
            "native_sent_tokens_est": nat_tok,
            "digest_tokens_est": dig_tok,
            "containment_ratio": round(raw_tok / dig_tok, 2) if dig_tok else None,
            "tokens_kept_out": max(0, raw_tok - dig_tok),
            "delta_vs_native_tokens": nat_tok - dig_tok,
            "raw_exact": self.raw_exact,
        }

    # -- bounded retrieval (omission without amnesia) ------------------------
    def get(self, stream: str = "stderr", lines: tuple[int, int] | None = None,
            span: str | None = None) -> str:
        """``ctx get run:<id>#<stream> --lines A:B`` — an exact bounded slice."""
        return _backend_get(self, stream, lines, span)

    def search(self, patterns, context: int = 0, fixed: bool = False) -> str:
        """``ctx search run:<id> <pattern>...`` — bounded search of the artifact."""
        if isinstance(patterns, str):
            patterns = [patterns]
        return _backend_search(self, list(patterns), context, fixed)

    def diff(self, later: "ContainedRun") -> str:
        """``ctx diff run:<before> run:<after>`` — regression delta digest."""
        return _backend_diff(self, later)


# ==============================================================================
# --- SANDBOX ---
# ==============================================================================

_slots: dict[int, str] = {}
_slot_lock = threading.Lock()


def _slot() -> str:
    """A short, stable directory name for the calling thread.

    Stable is the point. A unique per-task name (``bcb-4f2a91c0d8e3``) shows
    up twice in the captured bytes — in the manifest ``cwd`` and inside every
    traceback frame — so the same failing code rendered a *different* digest
    on every attempt. That silently defeats the prompt-prefix caching the
    harness is supposed to protect across a repair loop, and makes two arms
    solving the same task incomparable byte-for-byte.

    Runs are sequential, so one reusable slot per thread is enough isolation;
    :func:`new_sandbox` wipes it before handing it over.
    """
    tid = threading.get_ident()
    with _slot_lock:
        if tid not in _slots:
            _slots[tid] = f"w{len(_slots)}"
        return _slots[tid]


def new_sandbox(prefix: str = "task") -> Path:
    """A task sandbox *inside* the harness workspace.

    It must live under the workspace root: ``run_capture`` confines the cwd,
    and workspace-relative addresses only exist for paths inside it. The name
    is short and reused (see :func:`_slot`) so the digest stays reproducible
    and the traceback frame stays inside the evidence budget.

    ``SJ_KEEP_SANDBOX=1`` switches to unique per-task names so every sandbox
    survives for inspection. That is a debugging mode: it reintroduces the
    per-attempt path churn described above, so digests stop being comparable.
    """
    root = workspace_root()
    if os.environ.get("SJ_KEEP_SANDBOX") == "1":
        d = root / f"{prefix}-{uuid.uuid4().hex[:12]}"
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = root / _slot()
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def drop_sandbox(d: Path) -> None:
    if os.environ.get("SJ_KEEP_SANDBOX") == "1":
        return
    shutil.rmtree(d, ignore_errors=True)


# The digest renders one traceback line per failure and clips it at the
# upstream per-line evidence budget. Everything after the file path on that
# line -- `", line 42, in test_name"` -- is what makes the frame actionable,
# so the path has to leave room for it.
try:  # pragma: no cover - import guarded so status() works without ctx
    from ctx.textutil import EVIDENCE_LINE_CHARS as FRAME_LINE_BUDGET
except Exception:  # pragma: no cover
    FRAME_LINE_BUDGET = 160

# `File "` + `", line NNN, in some_test_method` for a realistic test name.
_FRAME_OVERHEAD_CHARS = 48


def frame_budget() -> dict:
    """How much of the innermost-frame line the sandbox path consumes.

    Reported in :func:`status` so a long ``SJ_WORKSPACE`` cannot quietly clip
    the line number out of every digest the way the in-repo default did.
    """
    probe = str(workspace_root() / _slot() / "prog.py")
    used = len(probe) + _FRAME_OVERHEAD_CHARS
    return {
        "sandbox_prog_path": probe,
        "frame_chars_used": used,
        "frame_chars_budget": FRAME_LINE_BUDGET,
        "frame_fits": used <= FRAME_LINE_BUDGET,
    }


# ==============================================================================
# --- PUBLIC ENTRY POINTS ---
# ==============================================================================

def contained_run(argv, *, cwd: Path | str, timeout: float = 120.0,
                  record_argv=None, env_extra: dict | None = None) -> ContainedRun:
    """Execute ``argv`` under the harness. Output never reaches the caller raw
    by accident — it is stored whole and returned as a bounded digest plus an
    addressable handle.

    ``cwd`` must be inside :func:`workspace_root`.
    """
    require()
    st = _resolve()
    if env_extra:
        # run_capture inherits the parent environment; MPLBACKEND=Agg and
        # friends have to be set here rather than passed through.
        for k, v in env_extra.items():
            os.environ[k] = str(v)
    if st["backend"] == "library":
        return _library_run(argv, cwd=cwd, timeout=timeout, record_argv=record_argv)
    return _cli_run(argv, cwd=cwd, timeout=timeout)


def contain_text(text: str, *, argv, exit_code: int = 1, stream: str = "stdout",
                 cwd: str = ".") -> ContainedRun:
    """Contain output that was produced outside the harness.

    Used for evaluators that synthesise a test log rather than executing one
    (SWE-bench Pro). The text is spooled into the store and digested by the
    *upstream* profile registry through the same ``ctx.invocation/v1``
    manifest the executor publishes — only the inputs are supplied here, no
    selection logic is re-implemented.
    """
    require()
    st = _resolve()
    if st["backend"] != "library":
        raise SJUnavailable(
            "contain_text needs the library backend (SJ_BACKEND=library); "
            "the CLI has no verb for ingesting out-of-band text."
        )
    return _library_contain_text(text, argv=argv, exit_code=exit_code,
                                 stream=stream, cwd=cwd)


# ==============================================================================
# --- LIBRARY BACKEND ---
# ==============================================================================

def _library_run(argv, *, cwd, timeout, record_argv) -> ContainedRun:
    from ctx.digest import render_run_digest
    from ctx.execution import ExecutionError, run_capture

    rel = _relativize(cwd)
    try:
        cap = run_capture(_ws, [str(a) for a in argv], cwd=rel, shell=False,
                          timeout=timeout, store=_store,
                          record_argv=list(record_argv) if record_argv else None)
    except ExecutionError as e:
        raise SJUnavailable(f"ctx run failed: {e}") from e

    digest, manifest = render_run_digest(_store, _ws, cap.manifest)
    return _from_manifest(digest, manifest)


def _library_contain_text(text: str, *, argv, exit_code: int, stream: str,
                          cwd: str) -> ContainedRun:
    from ctx.digest import render_run_digest
    from ctx.execution import invocation_manifest, stream_entries

    tmp = Path(tempfile.mkdtemp(prefix="sj-text-"))
    try:
        paths = {"stdout": tmp / "stdout", "stderr": tmp / "stderr"}
        for name, p in paths.items():
            p.write_text(text if name == stream else "", encoding="utf-8")
        streams = stream_entries(_store, paths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    manifest = invocation_manifest(
        _ws, cwd=cwd, argv=[str(a) for a in argv], shell=False,
        exit_code=exit_code, signal=None, timed_out=False, streams=streams,
    )
    manifest_id = _store.put_manifest(manifest, kind="run")
    manifest["id"] = f"sha256:{manifest_id}"
    digest, manifest = render_run_digest(_store, _ws, manifest)
    return _from_manifest(digest, manifest)


def _from_manifest(digest: str, manifest: dict) -> ContainedRun:
    from ctx.textutil import short_id

    short = short_id(manifest["id"])
    m = _DIGEST_HEADER_RE.search(digest or "")
    profile = m.group(2) if m else str(manifest.get("digest", {}).get("profile", "?"))
    streams = manifest.get("streams", {})
    res = manifest.get("result", {})

    return ContainedRun(
        handle=f"run:{short}",
        short_id=short,
        exit_code=res.get("exitCode"),
        timed_out=bool(res.get("timedOut")),
        profile=profile,
        digest=digest,
        raw_exact=True,
        stdout_blob=str(streams.get("stdout", {}).get("blob", "")).removeprefix("sha256:"),
        stderr_blob=str(streams.get("stderr", {}).get("blob", "")).removeprefix("sha256:"),
        stdout_bytes=int(streams.get("stdout", {}).get("bytes", 0)),
        stderr_bytes=int(streams.get("stderr", {}).get("bytes", 0)),
        stdout_lines=int(streams.get("stdout", {}).get("lines", 0)),
        stderr_lines=int(streams.get("stderr", {}).get("lines", 0)),
        backend="library",
        _manifest=manifest,
    )


def _read_blob(blob_hash: str, size: int, nbytes: int | None) -> str:
    """Read a stored stream, optionally only its tail.

    Seeks rather than reading the whole object: the bytes exist in the store
    precisely so they do not have to exist in this process.
    """
    try:
        path = _store.blob_path(_store.resolve_id(blob_hash, kinds=("blob",)))
        with path.open("rb") as fh:
            if nbytes and size > nbytes:
                fh.seek(size - nbytes)
            return fh.read().decode("utf-8", "replace")
    except Exception:
        return ""


def _relativize(cwd) -> str:
    p = Path(cwd).resolve()
    root = workspace_root().resolve()
    if p == root:
        return "."
    try:
        return str(p.relative_to(root))
    except ValueError as e:
        raise SJUnavailable(
            f"sandbox {p} is outside the straitjacket workspace {root}"
        ) from e


# ==============================================================================
# --- CLI BACKEND ---
# ==============================================================================

def _ctx_cli(args, timeout: float = 300.0) -> subprocess.CompletedProcess:
    exe = shutil.which("ctx")
    if not exe:
        raise SJUnavailable("`ctx` not found on PATH")
    return subprocess.run(
        [exe, "--workspace", str(workspace_root()), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _cli_run(argv, *, cwd, timeout) -> ContainedRun:
    rel = _relativize(cwd)
    out = _ctx_cli(["run", "--cwd", rel, "--timeout", str(timeout), "--",
                    *[str(a) for a in argv]], timeout=timeout + 60)
    digest = (out.stdout or "").strip()
    if not digest:
        raise SJUnavailable(f"ctx run produced no digest (rc={out.returncode}): "
                            f"{(out.stderr or '').strip()[:400]}")
    m = _DIGEST_HEADER_RE.search(digest)
    if not m:
        head = digest.splitlines()[0] if digest else ""
        raise SJUnavailable(f"unrecognised ctx digest header: {head[:200]!r}")
    short, profile = m.group(1), m.group(2)

    exit_code, stdout_b, stderr_b, stdout_l, stderr_l = _parse_digest_facts(digest)
    run = ContainedRun(
        handle=f"run:{short}", short_id=short, exit_code=exit_code,
        timed_out="timedOut" in digest, profile=profile, digest=digest,
        raw_exact=False, stdout_bytes=stdout_b, stderr_bytes=stderr_b,
        stdout_lines=stdout_l, stderr_lines=stderr_l, backend="cli",
    )
    # Recover the tail of each stream through bounded retrieval — enough for
    # the native-baseline arms under the configured cap, and honestly flagged
    # as inexact.
    # Recover enough to serve any native cap the caller might set; the payload
    # is truncated afterwards by native_payload(), not here.
    for name, nlines in (("stdout", stdout_l), ("stderr", stderr_l)):
        if not nlines:
            continue
        want = max(1, min(nlines, 200))
        got = _backend_get(run, name, (max(1, nlines - want + 1), nlines), None)
        run._recovered[name] = _strip_retrieval_header(got)
    return run


_DIGEST_STREAM_RE = re.compile(
    r"^(stdout|stderr):\s+([\d,]+) lines? · ([\d.,]+)\s*(B|KiB|MiB|GiB)", re.MULTILINE)
_DIGEST_EXIT_RE = re.compile(r"^exit:\s*(-?\d+)", re.MULTILINE)
_UNIT = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}


def _parse_digest_facts(digest: str):
    exit_code = None
    m = _DIGEST_EXIT_RE.search(digest)
    if m:
        exit_code = int(m.group(1))
    sizes = {"stdout": (0, 0), "stderr": (0, 0)}
    for sm in _DIGEST_STREAM_RE.finditer(digest):
        name = sm.group(1)
        lines = int(sm.group(2).replace(",", ""))
        nbytes = int(float(sm.group(3).replace(",", "")) * _UNIT[sm.group(4)])
        sizes[name] = (nbytes, lines)
    return (exit_code, sizes["stdout"][0], sizes["stderr"][0],
            sizes["stdout"][1], sizes["stderr"][1])


def _strip_retrieval_header(text: str) -> str:
    """Drop ``[ctx get …]`` / ``selector:`` framing and the ``Lnn: `` prefixes
    so the recovered text reads like the original stream."""
    out = []
    for ln in text.splitlines():
        if ln.startswith("[ctx ") or ln.startswith("selector:") or ln.startswith("coverage:"):
            continue
        out.append(re.sub(r"^L\d+: ?", "", ln))
    return "\n".join(out)


# ==============================================================================
# --- RETRIEVAL DISPATCH ---
# ==============================================================================

def _backend_get(run: ContainedRun, stream: str, lines, span) -> str:
    st = _resolve()
    ref = f"{run.handle}#{stream}"
    if st["backend"] == "library":
        from ctx.retrieval import Selector, get
        sel = Selector(lines=tuple(lines) if lines else None, span=span)
        return get(_store, _ws, ref, sel)
    args = ["get", ref]
    if lines:
        args += ["--lines", f"{lines[0]}:{lines[1]}"]
    if span:
        args += ["--span", span]
    return (_ctx_cli(args).stdout or "").rstrip("\n")


def _backend_search(run: ContainedRun, patterns, context: int, fixed: bool) -> str:
    st = _resolve()
    if st["backend"] == "library":
        from ctx.retrieval import search
        return search(_store, _ws, run.handle, patterns, fixed=fixed,
                      mode_all=False, context=context, glob=None, scope=None,
                      max_matches=None)
    args = ["search", run.handle, *patterns, "--context", str(context)]
    if fixed:
        args.append("--fixed")
    return (_ctx_cli(args).stdout or "").rstrip("\n")


def _backend_diff(before: ContainedRun, after: ContainedRun) -> str:
    st = _resolve()
    if st["backend"] == "library":
        from ctx.rundiff import run_diff
        return run_diff(_store, _ws, before.handle, after.handle)
    return (_ctx_cli(["diff", before.handle, after.handle]).stdout or "").rstrip("\n")


# ==============================================================================
# --- CLI SELF-CHECK ---
# ==============================================================================

# Two failure shapes, because they give opposite answers and only showing the
# flattering one would be the same overstatement this integration exists to
# remove. unittest writes its failure report to stderr and the code under test
# prints to stdout, so:
#
#   stderr-heavy  many failing tests with large assertion diffs. The untreated
#                 path forwards that flood; the digest replaces it. Containment
#                 wins here.
#   stdout-heavy  a chatty solution, one small failure. The untreated path
#                 never forwarded stdout, so there is nothing to win — the
#                 digest can even be slightly larger.
_SHAPES = {
    "stderr-heavy (big assertion diffs — the flood is in the failure report)": (
        "def task_func(n):\n    return [str(i) for i in range(n)]\n\n"
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        + "".join(
            f"    def test_{i:02d}(self):\n"
            f"        self.assertEqual(task_func(60), list(range(60)))\n"
            for i in range(12)
        )
        + DETERMINISTIC_UNITTEST_TAIL
    ),
    "stdout-heavy (chatty solution, one small failure)": (
        "def task_func(x):\n    return x + 5\n\n"
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_a(self): self.assertEqual(task_func(1), 2)\n"
        "for i in range(2000): print('noise %d' % i)\n"
        + DETERMINISTIC_UNITTEST_TAIL
    ),
}


def _selfcheck() -> int:
    print(json.dumps(status(), indent=2))
    if not available():
        print("\n" + INSTALL_HINT, file=sys.stderr)
        return 1

    for label, program in _SHAPES.items():
        print("\n" + "=" * 78)
        print(label)
        print("=" * 78)
        d = new_sandbox("selfcheck")
        try:
            (d / "prog.py").write_text(program, encoding="utf-8")
            run = contained_run([sys.executable, "prog.py"], cwd=d,
                                record_argv=["python3", "prog.py"],
                                env_extra=CAPTURE_ENV)
            m = run.metrics()
            print("\n--- digest ---")
            print(run.digest)
            print("\n--- metrics ---")
            print(json.dumps(m, indent=2))
            delta = m["delta_vs_native_tokens"]
            verdict = (
                f"digest is {delta:,} tokens SMALLER than the untreated baseline"
                if delta > 0 else
                f"digest is {abs(delta):,} tokens LARGER than the untreated baseline"
                if delta < 0 else "digest and untreated baseline are the same size"
            )
            print(f"\nA/B: {verdict} "
                  f"(baseline reads {m['native_stream']}: {m['native_sent_tokens_est']:,} tok "
                  f"· digest: {m['digest_tokens_est']:,} tok)")
            print(f"Captured but never resident: {m['tokens_kept_out']:,} tok — attested by "
                  f"the coverage receipt and retrievable by address, NOT the same number as "
                  f"the A/B above.")
            print("\n--- bounded retrieval (stderr 1:8) ---")
            print(run.get("stderr", (1, 8)))
        finally:
            drop_sandbox(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
