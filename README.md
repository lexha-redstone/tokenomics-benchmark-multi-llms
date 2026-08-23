# Multi-LLM Benchmark Suite for Tokenomics & Straitjacket

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Reports](https://img.shields.io/badge/Reports-indexed%20run%20log-purple.svg)](reports/README.md)
[![Straitjacket](https://img.shields.io/badge/Harness-ctx--harness%200.35.1-emerald.svg)](docs/straitjacket-implementation.md)

**Question this repository answers:** when a coding agent's tests fail, how
should the failure reach the next model — and what does each answer cost?

It benchmarks multi-LLM collaboration architectures (cascades, advisor/executor
splits, escalation ladders) against single frontier models on real software
engineering tasks, holding everything constant except one variable: how failing
test output is delivered to the repair turn. Models run live on Google Cloud
Vertex AI (Gemini) and Anthropic (Claude).

**New here? Read in this order.**

| | |
|---|---|
| 1 | [Key takeaways & best setting](#1-key-takeaways--best-setting) — what the runs found |
| 2 | [Why cascades suit BigCodeBench-Hard](#why-the-cascade-shape-suits-this-dataset) — which architecture pattern wins, and the mechanism behind it |
| 3 | [Run a benchmark](#3-run-a-benchmark) — exact files and commands |
| 4 | [Pipeline architecture](docs/pipeline-architecture.md) — how a task flows through the system |
| 5 | [Straitjacket implementation](docs/straitjacket-implementation.md) — what it is, and how it differs from upstream |
| 6 | [Report index](reports/README.md) — every sweep, in execution order |
| 7 | [Routing study](docs/routing-study.md) — finding the best gemini-3.7 + opus-5 combination |
| 8 | [Dataset selection for pattern tests](docs/pattern-dataset-selection.md) — where a non-cascade pattern could win, and which datasets can show it |

---

## 1. Key takeaways & best setting

From the largest sweep — **BigCodeBench-Hard, N=100**, live API
(`gemini-3.7-flash`, `claude-sonnet-5`, `claude-opus-5`), archived at
`bigCodeBench-hard/results/archive/bcb_n100_instrumented_20260822T2129.json`:

| Configuration | Pass rate | Total cost | **$ / solved task** |
|---|---|---|---|
| Single: `claude-opus-5` | **76%** | $3.52 | $0.0463 |
| Straitjacket Escalation Shield | 68% | $1.92 | **$0.0282** |
| Straitjacket Cascade | 66% | $2.23 | $0.0339 |
| Straitjacket Smart Repair | 64% | $2.72 | $0.0425 |
| Single: `gemini-3.7-flash` | 60% | $2.23 | $0.0372 |
| Straitjacket Hybrid | 59% | $1.63 | **$0.0277** |
| Single: `claude-sonnet-5` | 54% | $1.54 | $0.0285 |

### Best setting depends on what you are optimising

| If you want… | Use | Why |
|---|---|---|
| **Highest accuracy, cost no object** | `single_opus5` | 76% — nothing else reaches it |
| **Best accuracy per dollar** | `sj_escalation_shield` | **89% of Opus's pass rate at 61% of its cost per solved task**, and 55% of the absolute spend |
| **Cheapest working pipeline** | `sj_hybrid` | lowest $/solved at $0.0277; 59% pass rate |

The escalation shield (`gemini-3.5-lite` → `gemini-3.7-flash` → `claude-sonnet-5`,
each repair turn fed the contained digest) is the recommended default: it keeps
a frontier model in reserve for the hard tail without paying frontier prices on
the tasks a cheap model already solves.

### How much headroom is actually left

Comparing which *tasks* each arm solved, not just how many:

```
solved by gemini-3.7-flash but not claude-opus-5 :  3
solved by claude-opus-5 but not gemini-3.7-flash : 19
solved by both                                   : 57
solved by NEITHER                                : 21
────────────────────────────────────────────────────
perfect flash|opus router ceiling                : 79
union of all seven arms                          : 87
```

21 of the 100 tasks are out of reach for every model tested, so **79% is the
practical ceiling for a Flash/Opus router** — and Opus alone is already at 76%.
Routing is therefore mostly a *cost* play: the accuracy headroom is 3 points,
but Opus only needs to see ~40 of the 100 tasks. That observation is what the
[routing study](docs/routing-study.md) is built to exploit.

### Why the cascade shape suits this dataset

Three families of multi-model architecture were run over the same tasks. They
differ in **when** the extra model is paid for, relative to the first piece of
evidence about whether the task is actually hard:

| Pattern | When the extra model is spent | Arms |
|---|---|---|
| **Cascade / escalation** | *after* the tests fail — one attempt at a time, each next turn handed to a different model | `sj_cascade`, `sj_escalation_shield`, `sj_smart_repair` |
| **Planning & executing** | *before* anything runs — a planner writes guidance, a cheap executor implements it | `sj_hybrid`, `combo_read_write`, `G3` (N=50) |
| **Collaboration** | *before and across* — several candidates, then a synthesis turn reconciling them | `G4` dual-candidate verifier (N=50), `sj_dual_verifier` |

Every number below is recomputed from the raw result records by:

```bash
python3 tools/analyze_patterns.py
```

**The one measurement that is decisive.** Splitting each arm by which turn
solved the task isolates what the repair budget actually buys. `repair_loops=0`
means the first attempt passed, so the rescue rate is measured only over the
tasks that arm itself failed:

| Arm | First attempt | Model on 1st repair | Rescued by 1st repair | Model on 2nd repair | Rescued by 2nd |
|---|---|---|---|---|---|
| `sj_escalation_shield` | 31/100 | flash ↑ from lite | **28/69 = 41%** | `claude-sonnet-5` ↑ | 9/41 = 22% |
| `sj_cascade` | 32/100 | flash ↑ from lite | **28/68 = 41%** | flash again → | 6/40 = 15% |
| `sj_hybrid` (plan+exec) | 37/100 | flash ↑ from lite | 22/63 = 35% | — (no 2nd turn) | — |
| `sj_smart_repair` | 32/100 | lite ↓ **from flash** | **11/68 = 16%** | flash (medium) ↑ | 21/57 = 37% |
| `single_flash37` | 35/100 | flash → itself | 25/65 = 38% | — | — |
| `single_sonnet5` | 42/100 | sonnet → itself | 12/58 = 21% | — | — |
| `single_opus5` | 56/100 | opus → itself | 20/44 = 45% | — | — |

`sj_smart_repair` is the control that makes this readable: it is structurally a
cascade, but its first repair turn *de-escalates* (`gemini-3.7-flash` →
`gemini-3.5-flash-lite`). Its rescue rate collapses to 16% against the 41% of
the arms that escalate upward — **z = +3.55, p = 0.0004** — and recovers to 37%
on the next turn, the moment the ladder points up again. The repair turn's
rescue rate tracks the *capability of the model holding that turn*, and almost
nothing else.

**Two structural properties of BigCodeBench-Hard explain the rest.** Measured
over the same 100 tasks (`src/datasets.py`):

```
prompt fed to the model    mean 305 tok   median 278   max 863
gold solution              mean 174 tok   median 162   max 378
unit tests per task        mean   6       median   5   min   3
third-party libs per task  mean   3       median   3   max   6
```

1. **There is nothing to decompose.** Spec, solution and test suite together fit
   in under 2K tokens, and every task is a single function body. A planner
   cannot partition work that was never partitioned; on this dataset its output
   is largely a restatement of the docstring. That is why the planning premium
   lands almost entirely on the first attempt (37/100 for `sj_hybrid` vs 32/100
   for bare lite) and buys nothing afterwards.
2. **There is a free, exact oracle on every turn.** Roughly six executable unit
   tests per task return ground truth for $0. A cascade converts that oracle
   directly into a routing decision — *fail → escalate* — so the expensive model
   is bought only where the tests have already proved it is needed. A
   collaboration turn has to *replace* that oracle with a model's judgment about
   which candidate is better, which is a strictly weaker signal than simply
   running the tests. A planner spends before any oracle signal exists at all.

**The economics follow directly.** Counting actual calls to the premium model:

```
sj_hybrid   (plan & execute) : 163 gemini-3.7-flash calls  (100 planner, unconditional
                                + 63 repair)                → 59/100
sj_escalation_shield         :  69 gemini-3.7-flash calls
                                + 41 claude-sonnet-5 calls  → 68/100
                               (both conditional on a test failure)
```

The plan-and-execute arm bought 2.4× more premium turns than the escalation
shield and finished nine points lower, because 100 of its 163 premium calls were
committed against a prior rather than against evidence.

**What the data does *not* license.** Only the escalation-direction result above
clears significance at N=100. The pattern-level gaps do not: `sj_escalation_shield`
68% vs `sj_hybrid` 59% is p = 0.19; the first-repair 41% vs 35% is p = 0.42; the
planner's first-attempt lift (37 vs 32) is p = 0.46. The cascade arms also get
three attempts to `sj_hybrid`'s two, and that budget difference is a genuine
confound — comparing only the first repair turn is what controls for it. The
N=50 Gemini-vs-Claude sweep ranks the same way (escalation `C2` 38% / cascade
`G2` 36% > collaboration `G4` 34% > single-model repeat `G1` 30% > planning
`G3` 28%) but every one of those gaps is inside binomial noise at that size.

So the defensible claim is narrow and useful: **on BigCodeBench-Hard the return
comes from what the next turn escalates *to*, not from how much reasoning is
front-loaded before the first turn.** One suggestive detail from `G4` fits it —
its second candidate, an independent resample from the *same* model, rescued
**0 of 39** failures, while the stronger synthesis model that followed rescued 6.
Resampling changes the sample; escalation changes the ceiling.

### Caveats worth stating plainly

- **Containment is not free accuracy.** Replacing a paid triage model with the
  harness's digest removes 100% of triage spend ($0.0000 vs ~$0.0018/repair).
  It does not by itself raise pass rates; it lowers the cost of reaching them.
- **Containment does nothing when there is nothing to contain.** A failure whose
  whole output is a handful of lines shows a delta at or below zero, and the
  reports print that as readily as a win.
- **These are 100 tasks on one dataset.** Differences of a few points are within
  noise; the 76 vs 59 spread is not.
- **Report 11's BigCodeBench-Hard N=50 table does not reconcile with its own raw
  results.** It prints 30/47, 36/47, 37/47, 39/47 where the archived per-arm
  JSONs (`bigCodeBench-hard/results/n50_*.json`) record 15/50, 18/50, 14/50 and
  16/50. It also predates the *refuse rather than fabricate* failure policy in
  `src/client.py`, so its numbers may include simulated calls. The N=50 figures
  quoted above are read from those raw JSONs, not from the report.
- **The other two datasets here are not independent confirmation.** WebDev is a
  library-filtered *subset of BigCodeBench-Hard*
  ([src/datasets.py:125](src/datasets.py:125)), so it repeats the same task shape
  rather than testing a new one. SWE-bench Pro is not executed in this repo at
  all — `run_swebench_pro_task` scores a patch by substring-matching it against
  the canonical patch's first added lines
  ([src/evaluator.py:361](src/evaluator.py:361)), and the local dataset file has
  empty `FAIL_TO_PASS`/`PASS_TO_PASS`. Its numbers are proxy scores, not pass
  rates. Earlier sweeps
  ([report 11](reports/11_synthesis_cross-dataset.md)) report "the same shape" on
  both; that claim rests on one dataset and one proxy.

Where a *non*-cascade pattern should win, and which datasets could show it, is
worked out in
[dataset selection for pattern tests](docs/pattern-dataset-selection.md).

---

## 2. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # includes ctx-harness, the straitjacket harness
```

```bash
gcloud auth application-default login
export GCP_PROJECT="your-gcp-project-id"
export GCP_LOCATION="global"           # or us-central1
```

Verify the harness before spending money on a sweep:

```bash
python3 -m src.straitjacket
```

It prints backend status, runs two real captures, and performs a live bounded
retrieval. Model IDs and pricing live in [`MODELS.md`](MODELS.md).

---

## 3. Run a benchmark

**One entry point for every dataset:** [`run_benchmark.py`](run_benchmark.py).
The per-dataset directories contain data, results and historical scripts — you
do not need to run anything inside them.

### BigCodeBench-Hard

The dataset ships in the repository
(`bigCodeBench-hard/data/BigCodeBench-Hard-v0.1.4.jsonl`), so no download step
is needed.

```bash
# Smoke test: 5 tasks, one cheap variant. Confirms credentials and the harness.
python3 run_benchmark.py --dataset bcb --n 5 --variants single_flash_lite --report
```

```bash
# The headline sweep (report 12 was produced by this command).
python3 run_benchmark.py \
    --dataset bcb \
    --n 100 \
    --variants single_flash37,single_sonnet5,single_opus5,sj_cascade,sj_hybrid,sj_smart_repair,sj_escalation_shield \
    --report \
    --no-cache
```

```bash
# The ablation: identical models and prompts, only the evidence treatment differs.
# These are the rows that license any claim about containment.
python3 run_benchmark.py --dataset bcb --group ablation --n 30 --report
```

```bash
# Routing study: gemini-3.7-flash centric ladders with claude-opus-5 reserved
# for hard tasks. Ten arms; see docs/routing-study.md for what each one asks.
python3 run_benchmark.py --dataset bcb --group router --n 50 --report --no-cache
python3 tools/analyze_router_study.py
```

**Use `--no-cache` whenever you care about the containment receipt.** Cached
task records from older revisions have no containment field, and a cached run
silently produces an empty receipt.

Only run one sweep at a time. Concurrent runs write the same results file,
cache and reports, and the last one to finish wins.

### ClassEval — the sub-task routing experiment

BigCodeBench-Hard cannot test whether routing *sub-tasks* by difficulty pays,
because one of its tasks is one function with one verdict. ClassEval can: a task
is a class of ~4 methods, the dataset labels each method's dependency structure,
and every method ships its own test class — so a pass is attributable to the
model that wrote that method. The reasoning is in
[dataset selection for pattern tests](docs/pattern-dataset-selection.md); the
arms are in [`src/classeval.py`](src/classeval.py).

**Run the preflight once, before any sweep.** Nine of the hundred classes cannot
be scored on a clean machine — a missing optional import, an undownloaded
corpus, gold written against NumPy 1.x, one flaky random-map task, and two whose
`methods_info` omits a method the class needs. Left in, they are charged to the
models; the last two are charged *only* to the per-method arms, which biases the
exact comparison being made.

```bash
python3 tools/classeval_preflight.py --write --repeat 3
```

It runs ClassEval's own gold solutions, and writes the tasks gold cannot pass to
`classeval/data/quarantine-test.json`, which the loader then honours. Tasks are
excluded, never edited, and each exclusion records its reason.

```bash
# Smoke test: 2 tasks, the hypothesis arm and the shape it has to beat.
python3 run_benchmark.py --dataset classeval --n 2 --variants ce_route_by_tier,ce_cascade
```

```bash
# The full comparison. Eight arms over the 91 scorable classes.
python3 run_benchmark.py --dataset classeval --group classeval --n 91 --report --no-cache
python3 tools/analyze_classeval.py
```

| Arm | What it is |
|---|---|
| `ce_single_lite` / `ce_single_flash` / `ce_single_sonnet` | one model writes the whole class, then repairs it |
| `ce_cascade` | **the shape to beat** — whole-class escalation, Lite → 3.7 low → 3.7 medium |
| `ce_plan_exec` | planner writes per-method contracts, a cheap executor writes the class |
| `ce_route_flat` | **the control** — per-method generation, every method to the same model |
| `ce_route_by_tier` | **the hypothesis** — each method routed by its labelled difficulty tier |
| `ce_plan_route` | contracts *and* difficulty routing |

`ce_route_flat` is not optional. Without it, a win for `ce_route_by_tier` cannot
be told apart from "writing the class method-by-method is better", which is a
different claim — `tools/analyze_classeval.py` refuses to bless the result if
the control is missing from the sweep. The tier labels come from the dataset's
own `dependencies` annotation, never from a guess made here.

### Other datasets

```bash
python3 run_benchmark.py --dataset swebench --group straitjacket --n 30 --report
python3 run_benchmark.py --dataset webdev   --group single       --n 10 --report
```

Read the SWE-bench Pro caveat in [section 1](#caveats-worth-stating-plainly)
before trusting anything that path prints.

### CLI reference

| Flag | Values | Meaning |
|---|---|---|
| `--dataset` / `-d` | `bcb`, `swebench`, `webdev`, `classeval` | which benchmark |
| `--n` | integer | number of tasks |
| `--variants` / `-v` | comma-separated ids | exactly which architectures to run |
| `--group` / `-g` | `all`, `single`, `combo`, `straitjacket`, `nextgen`, `ablation`, `router`, `classeval` | a preset family instead of explicit ids |
| `--no-cache` | flag | ignore cached task results and re-run live |
| `--report` / `-r` | flag | emit the markdown report and HTML dashboard |
| `--out` / `-o` | path | override the consolidated JSON path |

List every variant id:

```bash
python3 -c "import sys;sys.path.insert(0,'.');from src.architectures import VARIANT_REGISTRY as R;[print(f'{k:<28}{v[\"name\"]}') for k,v in R.items()]"
```

### Where the output lands

| Artifact | Path |
|---|---|
| Consolidated metrics | `bigCodeBench-hard/results/bcb_all_results.json` |
| Per-task cache | `bigCodeBench-hard/results/cache_bcb_master.json` |
| Markdown report | `reports/NN_<dataset>_<tag>_n<N>.md` |
| HTML dashboard | `reports/NN_<dataset>_<tag>_n<N>.html` |

Reports are append-only and numbered in execution order, so a new sweep never
overwrites an earlier one's evidence. After a run:

```bash
python3 tools/index_reports.py --apply     # adopt new reports, refresh reports/README.md
```

---

## 4. Repository map

```
.
├── run_benchmark.py                 # ← the entry point for every dataset
│
├── src/                             # shared core library
│   ├── config.py                    #   model ids, pricing, prompt roles
│   ├── client.py                    #   Vertex AI + Anthropic dispatch, retry, usage
│   ├── datasets.py                  #   dataset loaders
│   ├── straitjacket.py              #   the ONLY bridge to ctx-harness
│   ├── routing.py                   #   evidence-gated escalation (difficulty signal + gates)
│   ├── evaluator.py                 #   sandboxed execution + the Evidence contract
│   ├── architectures.py             #   variant registry + every pipeline
│   ├── classeval.py                 #   ClassEval arms: per-method routing + its control
│   └── reporter.py                  #   markdown report + HTML dashboard
│
├── docs/                            # ← explanations (methodology, not results)
│   ├── straitjacket-implementation.md
│   ├── pipeline-architecture.md
│   ├── routing-study.md
│   ├── pattern-dataset-selection.md
│   ├── bigcodebench-hard-sweetspot-methodology.md
│   └── webdev-sweetspot-methodology.md
│
├── reports/                         # ← results only, indexed by run order
│   ├── README.md                    #   the index: read this first
│   └── NN_<dataset>_<tag>_n<N>.{md,html}
│
├── tests/                           # contract tests for the straitjacket bridge
├── tools/                           # pattern analysis, report indexing, auditing, resume
│
├── bigCodeBench-hard/               # dataset 1: Python function completion
├── swebench_pro/                    # dataset 2: enterprise git patch resolution (see caveat, §1)
├── webdev/                          # dataset 3: a library-filtered BCB-Hard subset
├── classeval/                       # dataset 4: class generation, scored per method
│     data/ also holds quarantine-<split>.json, written by the preflight
│     each: data/  results/  + historical sweep scripts
│
└── .agents/skills/                  # agent skills (see §6)
```

---

## 5. Straitjacket

Every digest in every report is produced by the upstream
[`straitjacket`](https://github.com/vamsiramakrishnan/straitjacket) harness
(`ctx-harness`). Nothing in this repository re-implements its evidence
selection — profile detection, digest rendering, coverage receipts and bounded
retrieval are all upstream calls.

Candidate solutions are executed **through** the harness, so test output is
captured at the birth gate: stored whole, never returned as an unbounded blob.
One capture yields two payloads — the raw stream an uncontained arm sends, and
the bounded digest a contained arm sends — so an A/B isolates the treatment and
nothing else.

If the harness is missing, straitjacket-labelled arms refuse to run rather than
produce something digest-shaped.

👉 **[Full implementation notes and differences from upstream](docs/straitjacket-implementation.md)**

```bash
python3 -m src.straitjacket                        # self-check, both failure regimes
pytest tests/test_straitjacket_integration.py -q   # contract tests
```

---

## 6. Agent skills

Bundled skills in [`.agents/skills/`](.agents/skills/) that AI coding
assistants load on demand.

**[`straitjacket`](.agents/skills/straitjacket/SKILL.md)** — context
containment and exact span retrieval for noisy commands: `ctx run -- <cmd>`,
`ctx get run:<id>#stdout --lines A:B`, `ctx diff run:<a> run:<b>`.

**[`tokenomics-architect`](.agents/skills/tokenomics-architect/SKILL.md)** —
picks an architecture, model tier and thinking level for a given task class.
Routing matrix derived from the sweeps in `reports/`:

| Task class | Recommended architecture |
|---|---|
| A — multi-library / algorithmic | Straitjacket Smart Repair |
| B — enterprise repo bug / git patch | Straitjacket Ultra-Sweet Hybrid |
| C — web & middleware | Straitjacket Hybrid |
| D — large CI/CD regression batch | Smart Tiered Cascade |

Thinking budget: `LOW` (~2k–4k tokens) is the default sweet spot for test
assertion repair; escalate to `MEDIUM` only for algorithmic deadlocks.

---

## 7. Contributing

[`straitjacket_benchmark_contribution_guide.md`](straitjacket_benchmark_contribution_guide.md)
carries the charter and the pre-commit checklist — including the rules that
keep a benchmark row honest: name the treatment you actually applied, never
degrade the baseline to flatter the mechanism, and make sure the arm whose
treatment *is* the baseline reports a delta of exactly zero.

```bash
pytest tests/ -q
```
