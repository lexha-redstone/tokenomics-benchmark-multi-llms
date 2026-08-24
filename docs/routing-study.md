# Routing Study: gemini-3.7-flash + claude-opus-5 on BigCodeBench-Hard

> **Status: run, over the complete dataset.** All eleven arms were executed on
> all **148** BigCodeBench-Hard tasks on 2026-08-24 —
> [report 19](../reports/19_bcb-hard_straitjacket_n148.md), and
> **[§5 is the answer](#5-the-result-n148--the-complete-dataset)**. §1's framing
> numbers still come from the earlier N=100 seven-arm sweep
> ([report 12](../reports/12_bcb-hard_straitjacket_n100.md)) and are kept as the
> study's starting position; §5 supersedes them where they disagree, and says so.

**Question.** Using `gemini-3.5-flash-lite` and `gemini-3.7-flash` as the
workhorses — with `claude-opus-5` reserved for tasks the Gemini tiers cannot
solve — which combination gives the best accuracy, and which gives the best
accuracy per dollar?

---

## 1. What the existing data already told us

> Kept as the study's starting position, from before it ran. **Two of its three
> premises did not survive N=148** — see
> [§5](#5-the-result-n148--the-complete-dataset). The ceiling was higher (91%
> across eleven arms, and Opus alone reaches 84.5% given three rungs), and
> "routing is a cost play" turned out to be true for the *evidence* gate and
> false for the attempt-count ladder.

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

Neither model had been run at N=148 before; report 12's numbers are N=100 with a
two-attempt budget, so these are new baselines rather than a re-run — which is
why `r0b_opus5_solo`'s 84.5% here and `single_opus5`'s 76% in report 12 are not
in conflict. Two variables differ: the task set and the repair budget.

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
# Full slice, all eleven arms, the complete dataset. This is what report 19 ran.
python3 run_benchmark.py --dataset bcb --group router --n=148 --report --no-cache

# Reads bigCodeBench-hard/results/bcb_router_results.json by default -- the file
# `--group router` writes. Pass --results to point it elsewhere.
python3 tools/analyze_router_study.py
```

### Cost

**Measured**, from the N=148 run in [report 19](../reports/19_bcb-hard_straitjacket_n148.md):

| Arms | N=148 actual |
|---|---|
| `r0a` sonnet-5 solo | $2.86 |
| `r0b` opus-5 solo | $5.70 |
| `r1`–`r2` 3.7-flash × 2 thinking levels | **$11.87** — `medium` alone was $7.60 |
| `r4`–`r5` Gemini ladders | $11.45 |
| `r6`–`r10` Opus-gated ladders | $26.39 |
| **all eleven** | **$58.27** |

The pre-run estimate was ~$40–45, so budget roughly 1.3× any extrapolation from
a smaller slice. The overshoot is concentrated in the high-thinking Gemini arms
for the reason §5.1 gives — they emit far more output tokens than the frontier
model does.

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

## 5. The result (N=148 — the complete dataset)

Run on 2026-08-24 over **all 148 BigCodeBench-Hard tasks**, eleven arms, live
API — [report 19](../reports/19_bcb-hard_straitjacket_n148.md). Reproduce the
analysis with:

```bash
python3 tools/analyze_router_study.py
```

| Arm | Pass | Total $ | **$/solved** | Opus called on | Opus yield | Avg rungs |
|---|---|---|---|---|---|---|
| `r0b_opus5_solo` | 125/148 (84.5%) | $5.6972 | $0.0456 | — | — | 1.6 |
| `r6_opus_after_ladder` | **125/148 (84.5%)** | $5.6533 | $0.0452 | 29% | 47% | 2.3 |
| `r8_opus_after_2` | 125/148 (84.5%) | $5.8400 | $0.0467 | 35% | 56% | 1.9 |
| `r10_opus_fresh_solve` | 122/148 (82.4%) | $5.0849 | $0.0417 | 27% | 35% | 2.2 |
| `r9_opus_on_evidence` | 120/148 (81.1%) | **$4.2374** | **$0.0353** | 45% | 58% | 2.0 |
| `r5_gemini_think_ladder` | 112/148 (75.7%) | $7.5467 | $0.0674 | — | — | 2.0 |
| `r7_opus_after_1` | 112/148 (75.7%) | $5.5787 | $0.0498 | 59% | 59% | 1.6 |
| `r2_f37_medium` | 111/148 (75.0%) | $7.5978 | $0.0684 | — | — | 1.9 |
| `r4_gemini_ladder` | 108/148 (73.0%) | $3.9043 | $0.0362 | — | — | 2.0 |
| `r1_f37_low` | 106/148 (71.6%) | $4.2716 | $0.0403 | — | — | 2.0 |
| `r0a_sonnet5_solo` | 99/148 (66.9%) | $2.8619 | $0.0289 | — | — | 1.9 |

**Oracle ceiling across all eleven arms: 135/148 (91%).** 13 tasks were solved
by nothing. The N=100 slice put the flash|opus ceiling at 79%; on the full set,
with three-rung budgets, `r0b_opus5_solo` alone reaches 84.5%.

**Pareto frontier** — nothing beats these on both accuracy and value:

```
r6_opus_after_ladder    84%   $0.0452/solved   $5.6533 total
r10_opus_fresh_solve    82%   $0.0417/solved   $5.0849 total
r9_opus_on_evidence     81%   $0.0353/solved   $4.2374 total
r0a_sonnet5_solo        67%   $0.0289/solved   $2.8619 total
```

### 5.1 The finding that inverts the study's premise

**A high-thinking Gemini rung costs more than Claude Opus-5 and scores worse.**

```
r2_f37_medium   gemini-3.7-flash (medium) x3   111/148   $7.5978   6,513 avg output tok
r0b_opus5_solo  claude-opus-5 x3               125/148   $5.6972   1,221 avg output tok
```

That is **33% more money for 14 fewer solved tasks.** `r5`, the low→medium→high
thinking ladder, lands in the same place ($7.5467, 112/148). The whole premise
of a budget ladder is that the cheap tier is cheap; at `medium` thinking on this
dataset it is not, because it emits 5.3× the output tokens Opus does. Thinking
budget, not model tier, is the dominant cost term once it is turned up.

This reframes what the gates are actually saving. Read §5.2 with it in mind.

### 5.2 Where the escalation ladder saves nothing

`r6_opus_after_ladder` — the conservative shape, Opus only after every Gemini
rung has failed — reaches **exactly** `r0b_opus5_solo`'s 125/148, at
**99% of its cost per solved task**. Running the whole Gemini ladder first
bought nothing: not accuracy, not money. `r8` (escalate after two failures) is
the same 125/148 and slightly *more* expensive.

So the reserve-the-frontier-model intuition, which the N=100 sweep supported at
a two-attempt budget, **does not survive a three-rung budget on the full
dataset**. Given three attempts, Opus solves the tasks the ladder would have
reached anyway, and the ladder's rungs are pure overhead on the tasks it cannot.

### 5.3 Where it does save — and why the mechanism is not the one predicted

`r9_opus_on_evidence` is the arm that pays: **81.1% for $4.2374**, which is
96% of Opus-solo's pass rate for **26% less total spend** and the best
`$/solved` of any arm above 70%.

§5's prediction was that the evidence gate would "call Opus **earlier on fewer
tasks**". Half right, and the wrong half is the instructive one:

| | Opus called on | Avg rungs | Total $ |
|---|---|---|---|
| `r6_opus_after_ladder` | 29% of tasks | 2.3 | $5.6533 |
| `r9_opus_on_evidence` | **45% of tasks** | 2.0 | **$4.2374** |

The evidence gate escalates to Opus **more often, not less** — and is cheaper
anyway. The saving does not come from rationing the frontier model. It comes
from **not paying for the `gemini-3.7-flash (medium)` rung that was going to
fail**: firing the gate at rung 1 or 2 skips the most expensive tier on the
ladder, which §5.1 shows costs more per task than Opus does.

This also corrects the framing in the study's own success criterion. "A gate
that escalates *less often* with the same pass rate is strictly better" is
wrong as stated, because it prices only the escalations and not the rungs spent
before the gate can fire. The right criterion is total spend at matched pass
rate, and by that measure the gate that escalates *more* wins.

The gate is also selective in the way it was designed to be: 58% of the tasks
`r9` handed to Opus were solved, against 47% for `r6` and 35% for `r10`. It is
finding harder-but-reachable failures, not burning budget on the unreachable 13.

### 5.4 Repairing beats re-solving

`r10_opus_fresh_solve` hands Opus the problem from scratch rather than the
failed candidate. It solves 122/148 against `r6`'s 125, and its frontier yield
is the worst on the board — **35%, against `r6`'s 47% on a comparable call
volume (27% vs 29% of tasks)**. The failing candidate and its digest are worth
something to the repair turn; discarding them to avoid anchoring costs more
than the anchoring does. It is cheaper in total ($5.08 vs $5.65) only because a
fresh solve carries less context, not because it works better.

### 5.5 Run integrity, checked rather than assumed

Read straight out of `bcb_router_results.json`:

- **Backend `library`, `ctx-harness` 0.35.1, available.** Every arm ran through
  the real harness.
- **0 simulated tasks in all eleven arms.** No `--allow-simulation` fallback
  contaminated a row.
- **All eleven arms completed 148/148 tasks.** No arm is scored over a smaller
  set than another.
- **`r9_opus_on_evidence` carries `routing.degraded = true` on 11 of 148 tasks.**
  On those, at least one attempt produced a `text/v1` capture with no typed fact
  tier, so the gate read `shallow` ("no typed evidence") and declined to
  escalate. The gate was working on the other attempts of the same tasks — this
  is per-attempt blindness, not the `SJ_BACKEND=cli` global failure §4 warns
  about.

That last one biases `r9` toward *under*-escalation, so it is worth asking
whether it explains the 5-task gap to `r6`. **It does not:**

```
on the 11 degraded tasks:   r9 solved 3    r6 solved 3
r6 solved but r9 did not:   9 tasks, of which only 1 was degraded
r9 solved but r6 did not:   4 tasks
```

The two arms disagree on 13 tasks in both directions, and degradation touches
one of them. The gap is a real property of the gate, not an artifact of the
blind spots. The blind spots are still worth closing — a profile with no fact
tier should probably read as `unknown` and escalate, not as `shallow` and hold.

### 5.6 What to take from this

1. **Watch the thinking budget before the model tier.** `medium` on
   `gemini-3.7-flash` is not a budget option on this dataset.
2. **Evidence-gated escalation is the recommended router** — `r9`, at 96% of
   frontier accuracy for 74% of frontier spend.
3. **An attempt-count ladder in front of a frontier model is not worth its
   rungs** at a three-attempt budget. If the frontier model is going to be
   called anyway, calling it sooner is cheaper.
4. **Do not throw the failed candidate away** when escalating.

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
