# The Straitjacket Implementation in This Repository

This document explains exactly how [`straitjacket`](https://github.com/vamsiramakrishnan/straitjacket)
(PyPI: `ctx-harness`, CLI: `ctx`) is wired into this benchmark, what this
repository adds on top, and — most importantly — **what it deliberately does
not change**.

- Bridge module: [`src/straitjacket.py`](../src/straitjacket.py)
- Evidence contract: [`src/evaluator.py`](../src/evaluator.py)
- Contract tests: [`tests/test_straitjacket_integration.py`](../tests/test_straitjacket_integration.py)

---

## 1. The one rule

> Every digest in every report comes from the upstream `ctx` package.
> Nothing in this repository re-implements its evidence selection.

Profile detection, digest rendering, coverage receipts, span minting and
bounded retrieval are all upstream calls. This repository supplies inputs and
records measurements; it never decides *which lines are the evidence*.

`src/straitjacket.py` is the only module allowed to produce a digest. If it
cannot reach the harness, straitjacket-labelled arms raise `SJUnavailable`
rather than degrade to something digest-shaped.

---

## 2. What the original provides

The upstream invariant is a **birth-time gate**:

> Potentially unbounded output must be captured before it reaches the model,
> or rejected before execution.

`ctx run -- pytest -q` executes the command, streams stdout/stderr to disk as
immutable content-addressed blobs, and returns a small deterministic digest
with three properties a truncated log does not have:

| Property | What it means |
|---|---|
| **Profile-typed** | A registry picks `unittest/v1`, `pytest/v2`, `lint/v1`, … by shape, and extracts the evidence that profile declares required — failing identities, innermost frames, exception classes. |
| **Coverage receipt** | The digest states what it parsed and what it omitted (`parsed: 3,052/3,052 lines · omitted: 3,039`). Omission is counted, not silent. |
| **Retrieval addresses** | Every omitted region stays reachable: `ctx get run:<id>#stderr --lines 1280:1300`, `ctx search run:<id> 'MissingTenantError'`. |

The anti-pattern it replaces is *position- or keyword-based* selection:

```python
# Anti-pattern. Produces a shorter string, not a digest.
lines = [l for l in stderr.splitlines()
         if any(k in l for k in ["FAIL:", "AssertionError"])]
```

No coverage receipt, no address for what was dropped, and the one quiet
anomaly in the middle of a repetitive log is gone with no record that it ever
existed. Position is not relevance.

---

## 3. How this repository calls it

### 3.1 Two backends

| Backend | Mechanism | When |
|---|---|---|
| `library` | In-process `import ctx` → `run_capture` + `render_run_digest` + `retrieval.get/search` | Default. Exact, fastest, gives the stored bytes back for the uncontained baseline. |
| `cli` | Subprocess `ctx run` / `ctx get` / `ctx search` | The published CLI surface, used as-is. Raw-stream recovery goes back through bounded retrieval, so `raw_exact` is `False`. |
| `off` | — | Harness unavailable. Nothing fabricates a digest. |

Both backends run the same upstream code; `library` simply skips a process
boundary. Select with `SJ_BACKEND`.

### 3.2 The capture path

```
run_bigcodebench(problem, code)
  └─ sj.contained_run([python, "prog.py"], cwd=<sandbox>)
       └─ ctx.execution.run_capture(...)          # stdout/stderr → immutable blobs
          └─ ctx.digest.render_run_digest(...)    # upstream profile registry
             → Evidence(native_payload,
                        digest=<real digest>,
                        run=<addressable handle>)
```

### 3.3 One capture, two treatments

`Evidence` subclasses `str`, so existing call sites keep working while
straitjacket-aware ones reach further:

| Accessor | Payload | Used by |
|---|---|---|
| `str(evidence)` | the *uncontained* payload: the failing stream, tail-truncated at `SJ_RAW_CAP` | `native` and `llm` arms |
| `evidence.digest` | the *contained* payload: the real upstream digest | `straitjacket` arms |
| `evidence.run` | the addressable handle | the bounded-retrieval arm |

Both arms therefore observe **the same execution**. The comparison isolates
the treatment and nothing else, which is what licenses any claim the reports
make.

### 3.4 The three treatments

`src/architectures.py` names the benchmark's independent variable instead of
hard-coding it:

```python
_treat_error(err, "native")        # raw tail, $0, stays resident every later turn
_treat_error(err, "llm")           # cheap model rewrite, costs tokens + a round trip
_treat_error(err, "straitjacket")  # the harness's own digest, $0, no round trip
```

An arm's registry label must name the treatment it actually applies. Run the
identical pipeline under all three with `--group ablation`.

### 3.5 Bounded retrieval

`run_contained_retrieval_cascade` exercises the half a digest alone does not.
When the digest is genuinely insufficient, the model may spend **one** bounded,
local, $0 lookup against the frozen artifact:

```
CTX: ctx get run:8d8335db6848#stderr --lines 1280:1300
CTX: ctx search run:8d8335db6848 'MissingTenantError' --context 3
```

Clamped to 80 lines / 6 lines of context, so retrieval cannot become the second
flood. The handle the model types is ignored — retrieval is always served
against that failure's own artifact, so a hallucinated id cannot address
another run.

---

## 4. What this repository adds

None of these change upstream behaviour. They exist because a *benchmark* has
requirements a *harness* does not.

### 4.1 Deterministic capture

Artifact identity is the sha256 of the captured bytes, so anything a runner
prints that varies between identical executions mints a new artifact for the
same failure. Measured across four invocations of one identical failing
program:

```
Ran 12 tests in 0.114s   → run:b10e76190835
Ran 12 tests in 0.111s   → run:1af74d563f47
Ran 12 tests in 0.112s   → run:5a64972759e6
```

Three handles, one failure — while the rendered digest body was byte-identical
every time, because upstream *does* strip that noise from the rendering.
Upstream's guarantee is a stable digest for stable captured bytes; producing
stable bytes is the harness caller's job.

`sj.DETERMINISTIC_UNITTEST_TAIL` pins the elapsed-time digits (keeping the
`Ran N tests in ` shape the profile detects on), and `sj.CAPTURE_ENV` sets
`PYTHONHASHSEED=0` and `MPLBACKEND=Agg` for every captured child.

### 4.2 A short, stable sandbox path

The sandbox path is not private bookkeeping — Python prints it inside every
traceback, so it competes for the digest's 160-character per-line evidence
budget. With sandboxes under the repository (83 characters before anything is
appended), the innermost-frame row rendered as:

```
innermost frame stderr:L5: File "/Users/…/.straitjacket/workspace/sandbox/selfcheck-ffc60b8faa14/prog.py", line
```

`, line 6, in test_a` — the line number, the reason that row exists — was
clipped off. Sandboxes now live at `~/.cache/tokenomics-sj/ws/w0`, and
`sj.status()["frame_budget"]` reports the headroom and warns if an override
eats it. Slot names are short *and reused*, so identical failures produce
identical bytes.

### 4.3 Bounded reads: never re-flood the harness

`run_capture` streams to disk so a flood never sits in memory. An adapter that
reads it back defeats that. Measured on a 40.9 MB stdout capture:

| Operation | Peak heap |
|---|---|
| `native_payload()` (bounded tail) | ~0 MB |
| `metrics()` | ~0 MB |
| `raw_stdout` (deliberate full read; tests and debugging only) | 81.8 MB |

Everything on the per-task path is a bounded tail read against the store; even
deciding *which* stream carried the failure peeks at only 4 KB.

### 4.4 The containment receipt

Pass rate and dollars measure one turn. Containment is about residency, so
every sweep records four numbers that are easy to conflate and are kept apart:

| Number | Meaning |
|---|---|
| **Captured** | everything the execution produced; the store holds all of it |
| **Sent to model** | what this arm actually placed in the repair prompt |
| **Native baseline** | what the untreated path would have sent for the same failures |
| **Δ vs native** | the A/B advantage — `Native baseline − Sent` |

`Captured − Sent` is the larger, more flattering number and is **not** the A/B:
an untreated harness also discards streams it never reads. The difference is
that discarding is amnesia, while straitjacket's omissions are counted and
addressable.

Self-consistency check: the arm whose treatment *is* the baseline must report
`Δ = +0`.

### 4.5 Refusal over fabrication

A row labelled "Straitjacket contained digest ($0.00)" must have been produced
by the harness:

```bash
SJ_BACKEND=off python3 run_benchmark.py --dataset bcb --group combo --n 10   # runs
SJ_BACKEND=off python3 run_benchmark.py --dataset bcb --group sj    --n 10   # refuses
```

`_arm(sj_required=True)` raises before the first API call. A second guard,
`_check_instrumented`, flags any arm that repaired something without recording
the treatment — a missing measurement is not a zero.

### 4.6 Out-of-band containment

An evaluator that synthesises a test log rather than executing one still has to
contain it. `sj.contain_text()` routes such a log through the real profile
registry by spooling it into the store behind the same `ctx.invocation/v1`
manifest the executor publishes. Only the inputs are supplied; no selection
logic is re-implemented. Library backend only. No dataset in the repository
currently takes this path — every arm executes its candidate for real — so it
exists for evaluators added later.

---

## 5. Differences from the original, at a glance

| Aspect | Upstream `ctx` | This repository |
|---|---|---|
| Evidence selection | **owns it** | never touches it |
| Invocation | `ctx run -- <cmd>` from an agent session | `sj.contained_run()` in-process, or the CLI |
| Store location | `$XDG_STATE_HOME/ctx` | `~/.cache/tokenomics-sj/state`, so a sweep never pollutes a real session store |
| Workspace | the user's repository | a short, disposable sandbox root |
| Runner output | whatever the command emits | elapsed time pinned, `PYTHONHASHSEED=0` — for reproducible artifacts |
| Raw bytes | stay on disk | stay on disk; the bridge reads bounded tails |
| Retrieval | agent types `ctx get …` | same commands, offered to the model as a `CTX:` protocol and served against the run's own artifact |
| Accounting | `ctx gain` | per-arm containment receipt in every report |
| Missing harness | n/a | straitjacket arms refuse to run |

Nothing in the middle column was modified. The right column is entirely
additive.

---

## 6. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SJ_BACKEND` | `auto` | `library`, `cli`, or `off` |
| `SJ_HOME` | `~/.cache/tokenomics-sj` | harness home — short and outside the repo on purpose (§4.2) |
| `SJ_WORKSPACE` | `<SJ_HOME>/ws` | workspace root; sandboxes are reused slots (`w0`, `w1`, …) |
| `CTX_STATE_HOME` | `<SJ_HOME>/state` | artifact store |
| `SJ_RAW_CAP` | `2500` | the single native-truncation knob; `0` = the true uncapped flood |
| `SJ_KEEP_SANDBOX` | unset | `1` keeps every sandbox — debug only; unique paths make digests non-reproducible |

---

## 7. Verifying it yourself

```bash
python3 -m src.straitjacket                                # status, two failure shapes, live retrieval
pytest tests/test_straitjacket_integration.py -q           # contract tests
```

The self-check deliberately runs **both** regimes: a stderr-heavy failure where
the digest replaces a real flood, and a stdout-heavy one where the untreated
path never forwarded the noise anyway and containment buys nothing. Showing
only the flattering one would be the overstatement this integration exists to
remove.

The contract tests pin: upstream provenance, birth-gate capture, coverage
receipts and addresses, digest determinism, the traceback line number
surviving the evidence budget, bounded reads, the native baseline reading the
failing stream, receipt self-consistency, and — parametrised over the whole
variant registry — that every arm records the treatment it applied.
