# Routing Study: gemini-3.7-flash + claude-opus-5 on BigCodeBench-Hard

> **Status: designed and implemented, not yet run at size.** The arms, gates and
> analysis exist and a one-task smoke run went through
> ([report 13](../reports/13_bcb-hard_straitjacket_n1.md)), but no N=50 sweep has
> been executed. Every number below comes from the **N=100 seven-arm sweep**
> ([report 12](../reports/12_bcb-hard_straitjacket_n100.md)); nothing here is a
> result of this study yet.

**Question.** Using `gemini-3.5-flash-lite` and `gemini-3.7-flash` as the
workhorses — with `claude-opus-5` reserved for tasks the Gemini tiers cannot
solve — which combination gives the best accuracy, and which gives the best
accuracy per dollar?

---

## 1. What the existing data already tells us

From the N=100 sweep
(`bigCodeBench-hard/results/archive/bcb_n100_instrumented_20260822T2129.json`):

```
solved by gemini-3.7-flash but not claude-opus-5 :  3
solved by claude-opus-5 but not gemini-3.7-flash : 19   ← what escalation can win
solved by both                                   : 57
solved by neither                                : 21   ← unreachable
────────────────────────────────────────────────────
perfect flash|opus router ceiling                : 79
```

Three consequences shape the design:

1. **The ceiling is 79%, and Opus alone already reaches 76%.** Routing is
   therefore a *cost* play far more than an accuracy play — the accuracy
   headroom over Opus-alone is 3 points, but the cost headroom is large,
   because Opus only needs to see ~40 of the 100 tasks.
2. **21 tasks are out of reach** for both models. Any arm reporting above 79%
   on this slice would be a measurement error, not a breakthrough.
3. **"Survived the cheap ladder" is a strong hard-task signal.** In the
   cascade arm, tasks resolved at repair loop 0 or 1 passed **100%** of the
   time; tasks still failing at loop 2 passed only **15%**.

Also relevant: `sj_hybrid` (advisor/executor split) scored 59% against
`sj_cascade`'s 66% on identical models. On this dataset, **escalation ladders
beat advisor/executor splits**, so the study builds on ladders.

---

## 2. The two gate families

A gate answers one question: *may the frontier model be called now?*

**Attempt-count gates** escalate after K failed rungs. Simple, and the 15%
figure above shows they work — but they are late by construction: the budget
for rungs 1..K is spent before the gate can fire.

**Evidence gates** read the harness's typed extraction
(`ContainedRun.evidence_graph()` → the profile's own `extract()`), so they can
escalate on turn one when the failure already looks hard, and decline to
escalate on a one-line slip. They are free: the evidence graph is a by-product
of a capture that already happened.

`src/routing.py` classifies each failure into one of four levels:

| Level | Meaning | Escalate? |
|---|---|---|
| `shallow` | every failure class is `SyntaxError` / `ImportError` / `NameError` … — the candidate never really ran | no — cheap models fix these |
| `local` | one or two failing identities | no |
| `broad` | ≥ 3 distinct failing identities | **yes** |
| `stalled` | the same failing identities survived a repair turn | **yes** — the model is not converging |

`stalled` is the sharpest signal and the narrowest: a single failing test that
does not move after a repair turn says more than three that are still shifting.

---

## 3. The arms

Twelve arms, in four blocks. Every repair turn is fed the straitjacket contained
digest, so the routing variable is isolated from the evidence-treatment
variable.

### Block 0 — frontier single-model baselines (`R0a`, `R0b`)

What the study has to beat. Both run **three rungs**, the same repair budget as
every other arm in the group — the existing `single_sonnet5` / `single_opus5`
variants use two, so comparing against them would credit the ladders with an
extra attempt.

| Arm | Ladder |
|---|---|
| `r0a_sonnet5_solo` | claude-sonnet-5 × 3 |
| `r0b_opus5_solo` | claude-opus-5 × 3 |

Neither model has been run at N=148 before; report 12's numbers are N=100 with a
two-attempt budget, so these are new baselines rather than a re-run.

### Block A — what is a thinking token worth? (`R1`–`R3`)

Identical ladder, three thinking levels. Everything downstream depends on
knowing whether `medium` is worth 
its extra tokens over `low`.

| Arm | Ladder |
|---|---|
| `r1_f37_low` | 3.7-flash (low) × 3 |
| `r2_f37_medium` | 3.7-flash (medium) × 3 |

### Block B — the Gemini-only ceiling (`R4`–`R5`)

No Opus. These are the control arms: whatever `R6`–`R10` achieve above these
is what Opus actually bought.

| Arm | Ladder |
|---|---|
| `r4_gemini_ladder` | Lite → 3.7 (low) → 3.7 (medium) |
| `r5_gemini_think_ladder` | 3.7 (low) → (medium) → (high), no Lite tier |

### Block C — how should Opus enter? (`R6`–`R10`)

| Arm | Gate | Question it answers |
|---|---|---|
| `r6_opus_after_ladder` | all Gemini rungs failed | the conservative baseline |
| `r7_opus_after_1` | one failure | is aggressive escalation worth it? |
| `r8_opus_after_2` | two failures | the middle setting |
| `r9_opus_on_evidence` | evidence says hard | can the digest route better than a counter? |
| `r10_opus_fresh_solve` | all rungs failed, **Opus re-solves from scratch** | is repairing a dead end worse than abandoning it? |

`r10` exists because a candidate several models failed to repair is often the
wrong *approach*, not a nearly-right one — repairing it anchors Opus to that
approach.

---

## 4. Running it

```bash
# Live smoke test first — this code has never touched a real API. ~$3-4.
python3 run_benchmark.py --dataset bcb --group router --n=10 --report --no-cache
python3 tools/analyze_router_study.py

# Full slice, all eleven arms.
python3 run_benchmark.py --dataset bcb --group router --n=148 --report --no-cache
```

### Cost

Extrapolated from measured per-attempt costs in the N=100 sweep
($0.0135/attempt for 3.7-flash, $0.0097 for sonnet-5, $0.0251 for opus-5) and
the observed first-attempt failure rates (65% / 58% / 44%):

| Arms | N=148 estimate |
|---|---|
| `r0a` sonnet-5 solo | ~$3 |
| `r0b` opus-5 solo | ~$7 |
| `r1`–`r2` 3.7-flash × 2 thinking levels | ~$9 (medium carries extra headroom) |
| `r4`–`r5` Gemini ladders | ~$7 |
| `r6`–`r10` Opus-gated ladders | ~$18 |
| **all eleven** | **~$40–45** |

Treat that as an order of magnitude, not a quote: arms that escalate early pay
more per attempt but run fewer rungs, and the two partly cancel.

Cheaper subsets:

```bash
# Just the thinking-level axis (block A) — no frontier spend at all. ~$9.
python3 run_benchmark.py --dataset bcb --variants r1_f37_low,r2_f37_medium --n=148 --report --no-cache

# The core question only: Gemini ladder vs two gates vs the Opus baseline. ~$17.
python3 run_benchmark.py --dataset bcb \
    --variants r0b_opus5_solo,r4_gemini_ladder,r6_opus_after_ladder,r9_opus_on_evidence \
    --n=148 --report --no-cache
```

Then:

```bash
python3 tools/analyze_router_study.py
```

which prints pass rate, `$/solved`, **how often Opus was actually invoked and
what fraction of those it solved**, the Pareto frontier, and the oracle ceiling
for the arms present.

**The evidence gate needs the library backend.** The typed fact tier is read
from the in-process manifest, so with `SJ_BACKEND=cli` there is no evidence to
gate on and `r9` silently behaves like `r6`. The router detects that: it warns
on stderr and sets `routing.degraded = true` on every affected task, so a
degraded arm is visible in the results rather than published as a finding.

`--no-cache` matters: cached task records from an earlier revision carry no
routing trace, and a cached run silently produces empty frontier columns.

---

## 5. Reading the result

The study is answered by three numbers per arm:

- **pass rate** against the 79% flash|opus ceiling;
- **`$/solved`** against `r0b_opus5_solo`, the same-budget frontier baseline;
- **frontier yield** — of the tasks handed to Opus, how many it solved. A low
  yield means the gate is escalating tasks that are unreachable anyway (the 21),
  and the budget is being burned on the wrong set.

A gate that escalates *less often* with the *same* pass rate is strictly
better. Expect the honest outcome to be a trade: the evidence gate should call
Opus earlier on fewer tasks, and whether that nets out depends on how well
`broad`/`stalled` correlate with the 19-task payoff set.

Negative results belong in the report. If the evidence gate does not beat
`after_ladder`, that is a finding about the signal, and
`src/routing.py`'s thresholds (`BROAD_FAILURE_ITEMS`) are the thing to revisit.

---

## 6. Implementation

| Piece | Where |
|---|---|
| Difficulty signal and gates | [`src/routing.py`](../src/routing.py) |
| Typed evidence accessor | `ContainedRun.evidence_graph()` in [`src/straitjacket.py`](../src/straitjacket.py) |
| The parameterised ladder | `run_tiered_router()` in [`src/architectures.py`](../src/architectures.py) |
| Arm definitions | `VARIANT_REGISTRY`, category 6 |
| Analysis | [`tools/analyze_router_study.py`](../tools/analyze_router_study.py) |
| Tests | [`tests/test_routing.py`](../tests/test_routing.py) |

Every task records a `routing` trace — the rungs actually called, whether the
frontier tier was used, and what each gate saw when it decided — so a sweep can
be re-examined without re-running it.
