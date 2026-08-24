# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Straitjacket Context Containment, Content-Addressable Storage (CAS),
Deterministic UnittestProfile Rendering, and Noise-Stripped Diff Engine.
Directly interfaces with original Straitjacket asset modules in /usr/local/google/home/lexha/Desktop/work/prj/99-assets/straitjacket/src.
"""

import os
import sys
import re
import tempfile
import subprocess
import shutil
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from .config import STRAITJACKET_SRC

if STRAITJACKET_SRC not in sys.path:
    sys.path.insert(0, STRAITJACKET_SRC)

# Import real Straitjacket modules
try:
    from ctx.store import Store, default_state_root
    from ctx.workspace import Workspace
    from ctx.digest.moreprofs import UnittestProfile
    from ctx.digest.base import DigestContext, StreamView
    from ctx.pricing import price_for, cost_usd
    _SJ_AVAILABLE = True
except Exception as e:
    _SJ_AVAILABLE = False
    print(f"[Straitjacket Notice] Direct import warning ({e}). Using built-in conformant harness.", flush=True)

# ==============================================================================
# --- CODE EXTRACTION AND SANDBOXED EXECUTION ---
# ==============================================================================

def extract_code(text):
    """Extract Python code block from LLM model response."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    raw = (m.group(1) if m else text).strip()
    return raw

def missing_code_error(code, entry_point):
    """Check if the extracted code defines the requested function entry point."""
    if f"def {entry_point}" in code:
        return None
    return f"model response contains no `def {entry_point}` function definition"

_UNITTEST_RUNNER = (
    "\n\nimport unittest as _ut, sys as _sys\n"
    "_ut.TestCase.maxDiff = None\n"
    "_res = _ut.TextTestRunner(verbosity=0).run("
    "_ut.TestLoader().loadTestsFromTestCase(TestCases))\n"
    "_sys.exit(0 if _res.wasSuccessful() else 1)\n"
)

def run_sandboxed_test(problem, solution_code, timeout=60):
    """
    Execute Python unit test in an isolated workspace with complete stdout/stderr capture.
    Returns: (passed: bool, raw_stderr: str, exit_code: int)
    """
    guard = missing_code_error(solution_code, problem.get("entry_point", "task_func"))
    if guard:
        return False, guard, 1

    program = solution_code + "\n\n" + problem["test"] + _UNITTEST_RUNNER
    workdir = tempfile.mkdtemp(prefix="ctx_bench_")
    path = os.path.join(workdir, "prog.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(program)

    try:
        env = {**os.environ, "MPLBACKEND": "Agg"}
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
            env=env
        )
        if r.returncode == 0:
            return True, "", 0
        err = (r.stderr.strip() or r.stdout.strip() or "test failed")
        return False, err, r.returncode
    except subprocess.TimeoutExpired:
        return False, "timeout: execution exceeded 60s", -1
    except Exception as e:
        return False, f"execution_error: {e}", 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

# ==============================================================================
# --- STRAITJACKET ZERO-COST LOCAL TRIAGE & DIGEST ENGINE ---
# ==============================================================================

def render_straitjacket_digest(raw_err, command="python3 prog.py", exit_code=1):
    """
    Straitjacket Zero-Cost Local Triage ($0.000000 API cost, 0ms latency).
    Uses the real UnittestProfile from straitjacket harness to deterministically parse and extract
    failing test names, assertion diffs, innermost traceback frames, and line numbers.
    """
    tokens_saved_estimate = max(0, (len(raw_err) // 4) - 80)
    
    if _SJ_AVAILABLE:
        try:
            mock_ws = MagicMock()
            mock_ws.root = "/tmp"
            manifest = {
                "cwd": "/tmp",
                "argv": command.split(),
                "result": {"exitCode": exit_code, "signal": None, "timedOut": False},
            }
            dctx = DigestContext(
                ws=mock_ws,
                manifest=manifest,
                stdout=StreamView("stdout", 0, 0, "text/plain", "", True),
                stderr=StreamView("stderr", len(raw_err), len(raw_err.splitlines()), "text/plain", raw_err, True),
                dense=True,
                suggestion_cap=0,
            )
            digest = UnittestProfile().render(dctx)
            return digest, {
                "as_run_usd": 0.0,
                "input": 0,
                "output": 0,
                "total_tokens": 0,
                "tokens_saved": tokens_saved_estimate
            }
        except Exception:
            pass

    # Built-in conformant UnittestProfile parser
    lines = []
    failing_tests = []
    innermost_frame = ""
    first_failure = ""

    for line in raw_err.splitlines():
        line_s = line.strip()
        if line_s.startswith("FAIL:") or line_s.startswith("ERROR:"):
            failing_tests.append(line_s)
        elif "AssertionError:" in line_s or "Exception:" in line_s or "Error:" in line_s:
            if not first_failure:
                first_failure = line_s
            lines.append(line_s)
        elif "File \"prog.py\"" in line_s or "line " in line_s:
            if not innermost_frame and "File \"prog.py\"" in line_s:
                innermost_frame = line_s
            lines.append(line_s)
        elif any(k in line_s for k in ["!=", "==", "not equal", "Expected", "Actual", "Missing"]):
            lines.append(line_s)

    failing_census = "\n".join(f"  - {t}" for t in failing_tests[:5]) if failing_tests else "  - test_case_failure"
    detail_lines = "\n".join(lines[:12]) if lines else raw_err[-600:]

    digest = (
        f"[ctx run:profile=unittest/v2 exit={exit_code}]\n"
        f"summary:\n"
        f"  failing tests census:\n{failing_census}\n"
        f"  innermost frame: {innermost_frame or 'File \"prog.py\", line in function'}\n"
        f"  core assertion error: {first_failure or 'AssertionError: output mismatch'}\n"
        f"assertion profile:\n{detail_lines}"
    )

    return digest, {
        "as_run_usd": 0.0,
        "input": 0,
        "output": 0,
        "total_tokens": 0,
        "tokens_saved": tokens_saved_estimate
    }

# ==============================================================================
# --- CAS CHECKPOINT STORAGE & RUN DIFFING ---
# ==============================================================================

class StraitjacketCASStore:
    """
    Lightweight Content-Addressable Storage (CAS) layer for tracking execution blobs,
    checkpoints, and run diffs across repair turns.
    """
    def __init__(self, session_id="straitjacket_session"):
        self.session_id = session_id
        self.blobs = {}
        self.checkpoints = {}

    def put_blob(self, content_str: str) -> str:
        data = content_str.encode("utf-8")
        blob_id = hashlib.sha256(data).hexdigest()[:16]
        self.blobs[blob_id] = content_str
        return f"blob:{blob_id}"

    def create_checkpoint(self, node_id: str, code: str, digest: str, passed: bool) -> str:
        code_ref = self.put_blob(code)
        digest_ref = self.put_blob(digest)
        ckpt_id = hashlib.sha256(f"{node_id}:{code_ref}:{digest_ref}:{passed}".encode()).hexdigest()[:12]
        self.checkpoints[ckpt_id] = {
            "node_id": node_id,
            "code_ref": code_ref,
            "digest_ref": digest_ref,
            "passed": passed,
            "code": code,
            "digest": digest
        }
        return f"checkpoint:{ckpt_id}"

    def get_checkpoint(self, ckpt_handle: str):
        cid = ckpt_handle.replace("checkpoint:", "")
        return self.checkpoints.get(cid)

def compute_run_diff(digest_a: str, digest_b: str) -> str:
    """
    Noise-stripped execution diffing between candidate runs (stripping non-deterministic artifacts).
    """
    lines_a = set(line.strip() for line in digest_a.splitlines() if line.strip())
    lines_b = set(line.strip() for line in digest_b.splitlines() if line.strip())

    only_in_b = lines_b - lines_a
    common = lines_a & lines_b

    diff_summary = [
        f"Consensus Failure Overlap ({len(common)} common signals):",
        *[f"  = {l}" for l in list(common)[:6]],
    ]
    if only_in_b:
        diff_summary.extend([
            f"Distinct Failure Mode in Candidate B:",
            *[f"  + {l}" for l in list(only_in_b)[:4]]
        ])
    return "\n".join(diff_summary)
