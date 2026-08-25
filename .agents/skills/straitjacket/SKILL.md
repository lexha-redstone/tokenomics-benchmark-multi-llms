---
name: straitjacket
description: Use this skill when executing noisy CLI commands, test suites (e.g., pytest, cargo test, build logs), or long-running commands in an agent session, or when inspecting specific lines from previously captured tool outputs using the 'ctx' CLI harness. Prevents context window flooding, eliminates prompt prefix cache drift, enables exact span retrieval, and exposes a typed evidence graph that a router can escalate on.
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
  - Any command whose output a *later* turn will have to route on (see §6).
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
- Retrieval is *bounded on purpose*. When this repository offers retrieval to a model it clamps the request to **80 lines / 6 lines of context** and serves it against that failure's own artifact, ignoring the handle the model typed — so a hallucinated id cannot address a different run, and retrieval cannot become the second flood.

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

- **Prompt Prefix Caching:** Because `straitjacket` strips non-deterministic noise (locale, temp paths, timestamps, PIDs), identical test failures produce **byte-identical digests** — guaranteed when the captured bytes, focus query, profile version, and policy version are all unchanged. This prevents prompt prefix drift across multi-turn repair attempts, preserving prompt cache hit rates (upstream A/B: **96.5–98.1%**, versus 80.6–84.2% for a transcript-rewriting proxy).
- **Compression ratio is not the billing ratio.** Digests collapse floods **8×–151×** (small outputs correctly pass through at ~1×), but measured end-to-end savings in upstream's live A/Bs are **−30% billed tokens / −17% cost**. Do not restate the compression ratio as a cost reduction.
- **Diffing True Signal:** Always prefer `ctx diff run:A run:B` over manual inspection when checking whether a code edit resolved a specific test regression.

### 4.1 Stable digests need stable bytes — and that is the *caller's* job

Upstream guarantees a stable digest for stable captured bytes. It does **not** make your runner deterministic. Artifact identity is the sha256 of the captured bytes, so anything the command prints that varies between identical executions mints a new artifact for the same failure. Measured here across four invocations of one identical failing program:

```
Ran 12 tests in 0.114s   → run:b10e76190835
Ran 12 tests in 0.111s   → run:1af74d563f47
Ran 12 tests in 0.112s   → run:5a64972759e6
```

Three handles, one failure — while the rendered digest body stayed byte-identical, because upstream *does* strip that noise from the rendering. If you depend on artifact identity (caching, dedup, "is this the same failure as last turn?"), pin the noise at the source:

- pin runner-emitted elapsed time, keeping the shape the profile detects on (`Ran N tests in `);
- set `PYTHONHASHSEED=0` (hash randomisation reorders any set a candidate prints) and `MPLBACKEND=Agg` (a GUI backend writes run-to-run churn);
- reuse short, stable sandbox slot names rather than minting a unique temp dir per run.

### 4.2 The sandbox path is model-visible evidence, not bookkeeping

Python prints the working directory inside every traceback, so it competes for the digest's **160-character per-line evidence budget**. With sandboxes nested under a deep checkout (83 characters before anything was appended), the innermost-frame row rendered as:

```
innermost frame stderr:L5: File "/Users/…/.straitjacket/workspace/sandbox/selfcheck-ffc60b8faa14/prog.py", line
```

`, line 6, in test_a` — the line number, the entire reason that row exists — was clipped off. Keep the capture root short and outside the checkout (`~/.cache/<tool>/ws/w0`), and check the headroom before blaming the profile.

### 4.3 Never re-materialise what the harness spooled to disk

`run_capture` streams to disk so a flood never sits in memory; an adapter that reads it all back defeats the point. Measured on a 40.9 MB stdout capture:

| Operation | Peak heap |
|---|---|
| bounded tail read | ~0 MB |
| `metrics()` | ~0 MB |
| full `raw_stdout` read (tests/debugging only) | 81.8 MB |

Everything on a per-task path should be a bounded tail read; even deciding *which* stream carried the failure only needs a 4 KB peek.

---

## 5. Containment Accounting: report four numbers, not one

Pass rate and dollars measure one turn. Containment is about **residency** — how many turns those bytes would have stayed in the transcript. Four numbers are easy to conflate and must be kept apart:

| Number | Meaning |
|---|---|
| **Captured** | everything the execution produced; the store holds all of it |
| **Sent to model** | what this arm actually placed in the prompt |
| **Native baseline** | what the *untreated* path would have sent for the same failures |
| **Δ vs native** | the A/B advantage — `Native baseline − Sent` |

`Captured − Sent` is the larger, more flattering number and is **not** the A/B: an untreated harness also discards streams it never reads. The difference is that discarding is amnesia, while straitjacket's omissions are counted in a coverage receipt and remain retrievable by address.

Measured on BigCodeBench-Hard N=148 (all eleven arms, `ctx-harness` 0.35.1, `library` backend, estimated tokens):

```
captured 62,569–128,647 · sent 15,035–36,529 · native baseline 31,205–76,598
Δ vs native: +51% to +54%, every arm
```

Self-consistency check that catches most accounting bugs: **the arm whose treatment *is* the baseline must report `Δ = +0`.**

Where containment does nothing is equally reportable: a failure whose whole output is a handful of lines shows a delta at or below zero, and a run that only ever floods *stdout* while the untreated path forwards *stderr* buys nothing. Showing only the flattering regime is the overstatement this accounting exists to remove.

---

## 6. The digest is also a routing signal (typed evidence graph)

The digest's second, less obvious half is that the profile's `extract()` output is machine-readable: typed failing identities, failure classes, and `file:line` loci. Reading it costs **$0** — it is a by-product of a capture that already happened — and it lets a repair loop decide *when to escalate* instead of counting attempts.

The classification used here (`src/routing.py`), from the typed graph only:

| Level | Rule | Meaning |
|---|---|---|
| `shallow` | every failure class ∈ {Syntax, Indentation, Tab, Import, ModuleNotFound, Name}Error | the candidate never really ran; any cheap model fixes it |
| `local` | 1–2 distinct failing identities | one bug, keep it cheap |
| `broad` | ≥ 3 distinct failing identities | worth a stronger model now |
| `stalled` | the identical failing identity set survived a repair turn | the model is not converging — hand it over |

`broad`/`stalled` is the escalation trigger. On the full BigCodeBench-Hard dataset this gate reached **96% of the frontier model's pass rate for 74% of its spend** — and it did so by escalating **more** often (45% of tasks vs 29%) but **earlier**, skipping an expensive middle rung that was going to fail anyway.

**Two guardrails this needs, or the result is fiction:**

1. **The fact tier only exists on the in-process backend.** Under a CLI/subprocess backend there is nothing typed to read, every failure classifies as `shallow`, the gate never fires early, and the arm silently degrades into a plain attempt-count ladder. Detect it, warn, and set a `degraded` flag on the affected records — a row that did not test what its name says must not be quotable.
2. **No fact tier is not the same as an easy failure.** `text/v1` (nothing recognised the output as a test run) is treated as `shallow` here on purpose, but it is a *default*, not evidence. On the N=148 sweep 11 of 148 tasks hit it; audit that count before reading a gate's result.

---

## 7. Capture through a container (or any wrapper)

`ctx run` takes an argv, so a containerised test run goes through the birth gate unchanged — no second containment implementation, and the same `pytest/v*` profile does the extraction, which is what keeps §6 working there too:

```bash
ctx run -- docker exec -w /repo <container> bash -lc "pytest tests/test_x.py"
```

Two things to get right, both learned the expensive way:

- **Record a stable argv.** A container name carrying a pid (or any per-run token) makes every attempt digest as a *different command*. Record the logical command (`pytest tests/test_x.py`) rather than the literal wrapper invocation.
- **One container per task, not per attempt.** A repair ladder runs several attempts against the same repository; paying container start-up each time triples the dominant cost and makes the benchmark measure Docker instead of the models. Start once, reset the worktree between attempts, tear down after.

---

## 8. How This Repository Uses `ctx` (Benchmark Harness)

This benchmark suite does not shell out to `ctx` per benchmark task; it calls the
same upstream code in-process through `src/straitjacket.py`, which is the only
place allowed to produce a straitjacket digest here.

| Skill concept | Where it lands in this repo |
|---|---|
| `ctx run -- <cmd>` | `src.straitjacket.contained_run()` → `ctx.execution.run_capture` + `ctx.digest.render_run_digest` |
| the digest | `Evidence.digest` (what a contained arm sends to the model) |
| the raw output | `Evidence` as a plain string (what an uncontained arm sends) — read back from the artifact store, never from a pipe |
| the typed evidence graph | `ContainedRun.evidence_graph()` → `src/routing.py` `classify()` / `GATES` |
| `ctx get` / `ctx search` | `ContainedRun.get()` / `.search()`, exposed to the model as the `CTX:` protocol in `run_contained_retrieval_cascade` |
| `ctx diff run:A run:B` | `ContainedRun.diff()` |
| `ctx gain` | the per-arm **Context Containment Receipt** in every generated report |
| a log the evaluator synthesised rather than executed | `sj.contain_text()` — spooled into the store behind the same manifest, so the real profile registry still selects the evidence |

Configuration knobs (`docs/straitjacket-implementation.md` §6):

| Variable | Default | Meaning |
|---|---|---|
| `SJ_BACKEND` | `auto` | `library` (in-process, required for §6 gating), `cli`, or `off` |
| `SJ_HOME` | `~/.cache/tokenomics-sj` | harness home — short and outside the repo on purpose (§4.2) |
| `CTX_STATE_HOME` | `<SJ_HOME>/state` | artifact store, so a sweep never pollutes a real session store |
| `SJ_RAW_CAP` | `2500` | the **single** native-truncation knob; `0` = the true uncapped flood |
| `SJ_KEEP_SANDBOX` | unset | debug only — unique paths make digests non-reproducible |

### Four rules this repository enforces on top of the skill

1. **Never hand-roll a digest.** Selecting lines because they contain `FAIL:` or
   `AssertionError`, or slicing `stderr[-4000:]`, produces shorter output with no
   coverage receipt and no address for what was dropped. Position is not
   relevance. `pytest tests/test_straitjacket_integration.py` fails if that
   pattern reappears in the evaluator.
2. **Refuse rather than pretend.** If `ctx-harness` is not installed, arms that
   claim containment raise `SJUnavailable` instead of degrading to something
   digest-shaped. A benchmark row must credit the mechanism that actually ran.
   A second guard flags any arm that repaired something without recording which
   treatment it applied — a missing measurement is not a zero.
3. **Do not degrade the baseline to flatter the mechanism.** The uncontained arm
   gets the failing stream truncated once (`SJ_RAW_CAP`). Feeding it stdout
   chatter it never used to forward, or measuring its baseline with a different
   budget than it actually sends, makes containment look better than it is. The
   receipt reports `Δ vs native`, and the native arm's own delta must be `+0`.
4. **Containment is not free accuracy.** It removes 100% of triage spend
   ($0.0000 vs ~$0.0018/repair) and cuts what stays resident by ~52%. It does
   **not** by itself raise pass rates; it lowers the cost of reaching them.
   Claim residency and dollars; do not claim accuracy.

Verify any of this locally:

```bash
python3 -m src.straitjacket
```

```bash
pytest tests/test_straitjacket_integration.py -q
```

The self-check deliberately runs **both** regimes — a stderr-heavy failure where the digest replaces a real flood, and a stdout-heavy one where containment buys nothing.
