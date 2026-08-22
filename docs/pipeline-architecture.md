# Pipeline Architecture

How a single benchmark task travels from dataset row to scored result, and how
the seventeen registered architecture variants differ from one another.

---

## 1. The task loop

Every variant, on every dataset, runs the same skeleton. Only the *models* and
the *evidence treatment* change.

```
                       ┌──────────────────────────────────────────┐
   dataset row ───────►│ 1. PROMPT      _build_initial_prompt()    │
   (src/datasets.py)   │    solver / advisor role from config.py   │
                       └───────────────────┬──────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────────┐
                       │ 2. GENERATE    dispatch_model()          │
                       │    Vertex AI Gemini | Anthropic Claude   │
                       └───────────────────┬──────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────────┐
                       │ 3. EXTRACT     extract_code / _patch     │
                       │    guard: is the entry point defined?    │
                       └───────────────────┬──────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────────┐
                       │ 4. EXECUTE     ctx run  (birth gate)     │
                       │    stdout/stderr → immutable blobs       │
                       │    upstream profile → bounded digest     │
                       └───────────────────┬──────────────────────┘
                                           ▼
                                    ┌──────────────┐
                              pass  │  tests OK?   │  fail
                        ┌───────────┴──────────────┴────────────┐
                        ▼                                       ▼
                   record result            ┌──────────────────────────────────┐
                                            │ 5. TREAT   _treat_error(...)     │
                                            │    native | llm | straitjacket   │
                                            └───────────────┬──────────────────┘
                                                            ▼
                                            ┌──────────────────────────────────┐
                                            │ 6. REPAIR  dispatch_model()      │
                                            │    escalate model / thinking     │
                                            └───────────────┬──────────────────┘
                                                            │
                                            back to step 4, up to max_repairs
```

Step 4 is the load-bearing one. Candidate code is executed **through** the
straitjacket harness, so the test output is captured before it can reach a
model — see [straitjacket-implementation.md](straitjacket-implementation.md).

Step 5 is the benchmark's **independent variable**. Everything else is held
constant so the comparison means something.

---

## 2. Module map

| Module | Responsibility |
|---|---|
| [`src/config.py`](../src/config.py) | Model IDs, pricing table, prompt roles (solver / advisor / executor / repair / triage) |
| [`src/client.py`](../src/client.py) | `dispatch_model()` — Vertex AI + Anthropic, retry, thinking budgets, usage accounting |
| [`src/datasets.py`](../src/datasets.py) | Loaders for BigCodeBench-Hard, SWE-bench Pro, WebDev |
| [`src/straitjacket.py`](../src/straitjacket.py) | The only bridge to `ctx-harness`: capture, digest, bounded retrieval |
| [`src/evaluator.py`](../src/evaluator.py) | Sandboxed execution, patch verification, the `Evidence` contract, containment ledger |
| [`src/architectures.py`](../src/architectures.py) | The variant registry and every pipeline implementation |
| [`src/reporter.py`](../src/reporter.py) | Markdown TCO report + HTML dashboard, indexed report naming |
| [`run_benchmark.py`](../run_benchmark.py) | The unified CLI entry point |

Per-dataset directories (`bigCodeBench-hard/`, `swebench_pro/`, `webdev/`) hold
data, results, a thin runner adapter, and historical one-off sweep scripts.

---

## 3. The evidence treatments

Every architecture eventually has to put a failed test run in front of a
model. *How* is what this suite measures:

| Treatment | Payload the model sees | Cost | Residency |
|---|---|---|---|
| `native` | raw failing stream, tail-truncated at `SJ_RAW_CAP` (2500 chars) | $0 | full payload stays in the transcript every later turn |
| `llm` | a cheap model's rewrite of that log | input+output tokens, one round trip | short, but unattested and paraphrased |
| `straitjacket` | the harness's bounded digest | $0, no round trip | short, coverage-attested, omissions stay addressable |

Set it per call: `run_cascade(problem, error_treatment="straitjacket")`.
A variant's registry label must name the treatment it actually applies.

---

## 4. Architecture families

### Category 1 — Single models
One model, one shot, plus one repair turn on failure. The baseline everything
else is measured against.

`single_flash_lite` · `single_flash37` · `single_sonnet5` · `single_opus5`

### Category 2 — Model combinations, no containment
Multi-model pipelines whose repair turn is fed untreated or LLM-triaged output.

| Variant | Shape |
|---|---|
| `combo_read_write` | read-heavy **advisor** writes a contract → write-heavy **executor** writes code |
| `combo_cascade_llm` | cheap generator → escalate to a stronger model, raw output in the repair prompt |
| `combo_hybrid_llm` | advisor + executor, repair fed a **paid** LLM triage digest |

### Category 3 — Combinations + straitjacket
Same shapes, repair turn fed the contained digest instead.

| Variant | Shape |
|---|---|
| `sj_cascade` | 3.5-Lite generate → 3.7-Flash escalate, up to 2 repairs |
| `sj_hybrid` | Flash plan + Lite exec + Flash repair |
| `sj_escalation_shield` | Lite → Flash → **Claude Sonnet-5** as the last line |
| `sj_smart_repair` | Flash (low) → Lite → Flash (medium thinking), pure Google stack |
| `sj_ultra_sweet` | Sonnet-5 architect → Lite executor → Opus-5 repair |

### Category 4 — Next-gen multi-provider
`sj_dual_verifier` — four tiers, Lite → Flash → Sonnet-5 → Opus-5.

### Category 5 — Evidence-treatment ablation
**The rows that license any claim about containment.** Identical models,
identical prompts, identical evaluation; only step 5 differs.

| Variant | Treatment |
|---|---|
| `ablate_cascade_native` | `native` |
| `ablate_cascade_llm_triage` | `llm` |
| `ablate_cascade_straitjacket` | `straitjacket` |
| `sj_contained_retrieval` | `straitjacket` + one bounded `ctx get` / `ctx search` lookup |

```bash
python3 run_benchmark.py --dataset bcb --group ablation --n 30 --report
```

---

## 5. What each run records

Per task:

```json
{
  "task_id": "BigCodeBench/13",
  "passed": false,
  "as_run_usd": 0.0036,
  "triage_usd": 0.0,
  "output_tokens": 612,
  "repair_loops": 2,
  "containment": {
    "captures": 3,
    "raw_tokens_est": 8110,
    "digest_tokens_est": 459,
    "evidence_sent_tokens_est": 306,
    "native_baseline_tokens_est": 625,
    "delta_vs_native_tokens": 319,
    "treatments": ["straitjacket"],
    "profiles": ["unittest/v1"],
    "handles": ["run:30edbceaec1c", "..."]
  }
}
```

The `handles` are live: the artifacts stay in the store, so any run in any past
sweep can still be interrogated.

```bash
ctx get run:30edbceaec1c#stderr --lines 1:40
ctx search run:30edbceaec1c 'AssertionError' --context 3
```

Per variant, the runner aggregates these into the **Context Containment
Receipt** printed in every report — see §4.4 of
[straitjacket-implementation.md](straitjacket-implementation.md) for why
`Captured − Sent` and `Δ vs native` are deliberately kept apart.

---

## 6. Adding a variant

1. Implement the pipeline in `src/architectures.py`.
2. Take an `error_treatment` parameter; call
   `_treat_error(err, error_treatment, problem=problem, is_swe=is_swe)`.
   Never re-summarise, keyword-filter or tail-truncate a failure yourself.
3. Decorate with `@_arm()`, or `@_arm(sj_required=True)` if it claims
   containment.
4. Register it in `VARIANT_REGISTRY` with a `triage_mode` that names the
   treatment it actually applies.
5. `pytest tests/test_straitjacket_integration.py -q` — the registry-wide
   tests will pick the new variant up automatically.

Full checklist: [`../straitjacket_benchmark_contribution_guide.md`](../straitjacket_benchmark_contribution_guide.md).
