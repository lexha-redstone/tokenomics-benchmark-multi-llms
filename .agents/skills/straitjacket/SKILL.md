---
name: straitjacket
description: Use this skill when executing noisy CLI commands, test suites (e.g., pytest, cargo test, build logs), or long-running commands in an agent session, or when inspecting specific lines from previously captured tool outputs using the 'ctx' CLI harness. Prevents context window flooding, eliminates prompt prefix cache drift, and enables exact span retrieval.
---

# Straitjacket CLI Harness: Context Containment & Exact Span Retrieval

When running test suites, build tools, or verbose CLI commands, raw stdout/stderr can flood the agent's context window (e.g., a single `pytest -q` run can consume 10k–300k+ tokens). Re-sending these logs on every subsequent turn slows down the session, increases token costs, and risks losing critical error lines when the host compaction runs.

The `straitjacket` CLI (`ctx`) solves this by capturing raw bytes into an immutable local store and presenting a small, deterministic **digest** with exact retrieval addresses.

---

## 1. Graduated Engagement (When to Use vs. When NOT to Use)

- **ALWAYS USE (`ctx run -- <command>`) for:**
  - Test suite executions (`pytest`, `unittest`, `cargo test`, `go test`, `npm test`, `bazel test`).
  - Heavy compiler or build commands where verbose logs or stack traces are expected.
  - Commands likely to generate **> 1,000 tokens** of output.
- **DO NOT USE for:**
  - Short, highly targeted commands (`git status`, short `ls`, `whoami`, simple `pwd`).
  - Outputs expected to be **< 1,000 tokens** where direct inline reading is simpler and indirection adds unnecessary overhead.

---

## 2. Core Operational Workflow (3-Step Loop)

### Step 1: Capture & Digest
Instead of running a verbose command directly, wrap it with `ctx run`:
```bash
ctx run -- pytest -q
```
*Why:* The command executes normally, but the raw output streams into an immutable local store. The agent context receives a compact, deterministic digest (~200 tokens) instead of the raw flood.

### Step 2: Understand the 4-Part Deterministic Digest
The digest returned by `ctx run` conforms to a standard 4-part structure:
```
[ctx run:8d8335db6848 profile=pytest/v2]
command: pytest -q
exit: 1
stdout: 4,102 lines · 402.1 KiB · est 98,000 tokens
failing tests (census):
  1. tests/test_auth.py::test_token_expiry   tests/test_auth.py:42
coverage:
  census: 1/1 identities inline · attested complete
  shown: 1 spans · omitted: 4,098 lines
next:
  ctx get run:8d8335db6848#stdout --lines 1280:1300
```
- **Header & Stats:** Shows total lines, byte size, and the estimated token count of the **raw** captured output — i.e. what was kept out of the context window, not the size of the digest itself.
- **Failing Tests Census:** Lists every failing test with its exact `file:line` coordinates.
- **Assertion Profile / Shown Spans:** Displays the deterministic core assertion or error snippet with ephemeral noise (timestamps, PIDs, ANSI colors) removed.
- **Next (Span Address):** Provides an exact command to retrieve any omitted byte range.

### Step 3: Surgical Retrieval (On-Demand)
If you need more context around an error than shown in the initial digest, **do not re-run the test command**. Query the exact byte/line slice using the address from `next`:
```bash
# Retrieve a specific line range from the saved execution run
ctx get run:8d8335db6848#stdout --lines 1280:1300
```
- **Rule of Thumb:** Request at most **50–100 lines per retrieval** to maintain strict context window discipline.

---

## 3. Essential `ctx` Command Cheat Sheet

| Command | Purpose | Example Usage |
|---|---|---|
| `ctx run -- <cmd>` | Execute a command, capture raw stdout/stderr locally, and output a bounded digest. | `ctx run -- pytest tests/` |
| `ctx get run:<id>#<stream> --lines <start>:<end>` | Retrieve an exact line range from a stored execution run without re-execution. | `ctx get run:8d8#stdout --lines 45:90` |
| `ctx diff run:<id1> run:<id2>` | Compare two execution runs, stripping ephemeral noise to show true signal diffs. | `ctx diff run:8d8 run:9f2` |
| `ctx search <handle> <pattern...>` | Bounded multi-pattern search over one captured artifact or repo path. The handle is **required** and comes first; patterns are regex by default (`--fixed` for literals). Not a semantic/embedding search. | `ctx search run:8d8#stdout "AssertionError" "token"` |
| `ctx gain` | Cumulative token/cost savings accumulated from capture telemetry. | `ctx gain` |

---

## 4. Why Determinism & Prompt Cache Preservation Matter

- **Prompt Prefix Caching:** Because `straitjacket` strips non-deterministic noise (locale, temp paths, timestamps, PIDs), identical test failures produce **byte-identical digests** — guaranteed when the captured bytes, focus query, profile version, and policy version are all unchanged. This prevents prompt prefix drift across multi-turn repair attempts, preserving prompt cache hit rates (measured **96.5–98.1%**, versus 80.6–84.2% for a transcript-rewriting proxy).
- **Compression ratio is not the billing ratio.** Digests collapse floods **8×–151×** (small outputs correctly pass through at ~1×), but measured end-to-end savings in live A/Bs are **−30% billed tokens / −17% cost**. Do not restate the compression ratio as a cost reduction.
- **Diffing True Signal:** Always prefer `ctx diff run:A run:B` over manual inspection when checking whether a code edit resolved a specific test regression.

---

## 5. How This Repository Uses `ctx` (Benchmark Harness)

This benchmark suite does not shell out to `ctx` per benchmark task; it calls the
same upstream code in-process through `src/straitjacket.py`, which is the only
place allowed to produce a straitjacket digest here.

| Skill concept | Where it lands in this repo |
|---|---|
| `ctx run -- <cmd>` | `src.straitjacket.contained_run()` → `ctx.execution.run_capture` + `ctx.digest.render_run_digest` |
| the digest | `Evidence.digest` (what a contained arm sends to the model) |
| the raw output | `Evidence` as a plain string (what an uncontained arm sends) — read back from the artifact store, never from a pipe |
| `ctx get` / `ctx search` | `ContainedRun.get()` / `.search()`, exposed to the model as the `CTX:` protocol in `run_contained_retrieval_cascade` |
| `ctx diff run:A run:B` | `ContainedRun.diff()` |
| `ctx gain` | the per-arm **Context Containment Receipt** in every generated report |

Two rules this repository enforces on top of the skill:

1. **Never hand-roll a digest.** Selecting lines because they contain `FAIL:` or
   `AssertionError`, or slicing `stderr[-4000:]`, produces shorter output with no
   coverage receipt and no address for what was dropped. `pytest
   tests/test_straitjacket_integration.py` fails if that pattern reappears.
2. **Refuse rather than pretend.** If `ctx-harness` is not installed, arms that
   claim containment raise `SJUnavailable` instead of degrading to something
   digest-shaped. A benchmark row must credit the mechanism that actually ran.
3. **Do not degrade the baseline to flatter the mechanism.** The uncontained arm
   gets the failing stream truncated once (`SJ_RAW_CAP`). Feeding it stdout
   chatter it never used to forward, or measuring its baseline with a different
   budget than it actually sends, makes containment look better than it is. The
   receipt reports `Δ vs native`, and the native arm's own delta must be `+0`.
