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
| 2 | [Run a benchmark](#3-run-a-benchmark) — exact files and commands |
| 3 | [Pipeline architecture](docs/pipeline-architecture.md) — how a task flows through the system |
| 4 | [Straitjacket implementation](docs/straitjacket-implementation.md) — what it is, and how it differs from upstream |
| 5 | [Report index](reports/README.md) — every sweep, in execution order |

---

## 1. Key takeaways & best setting

From the largest sweep, **[report 12 — BigCodeBench-Hard, N=100](reports/README.md)**
(`gemini-3.7-flash`, `claude-sonnet-5`, `claude-opus-5`, live API):

| Configuration | Pass rate | Total cost | **$ / solved task** |
|---|---|---|---|
| Single: `claude-opus-5` | **72%** | $3.61 | $0.0501 |
| Straitjacket Escalation Shield | 64% | $1.81 | **$0.0283** |
| Straitjacket Cascade | 62% | $2.18 | $0.0351 |
| Single: `gemini-3.7-flash` | 62% | $2.29 | $0.0369 |
| Straitjacket Smart Repair | 62% | $2.79 | $0.0449 |
| Straitjacket Hybrid | 59% | $1.62 | **$0.0275** |
| Single: `claude-sonnet-5` | 53% | $1.57 | $0.0297 |

### Best setting depends on what you are optimising

| If you want… | Use | Why |
|---|---|---|
| **Highest accuracy, cost no object** | `single_opus5` | 72% — nothing else reaches it |
| **Best accuracy per dollar** | `sj_escalation_shield` | 89% of Opus's pass rate at **56% of its cost per solved task**, and half the absolute spend |
| **Cheapest working pipeline** | `sj_hybrid` | lowest $/solved at $0.0275; 59% pass rate |

The escalation shield (`gemini-3.5-lite` → `gemini-3.7-flash` → `claude-sonnet-5`,
each repair turn fed the contained digest) is the recommended default: it keeps
a frontier model in reserve for the hard tail without paying frontier prices on
the 60% of tasks a cheap model already solves.

### Findings that hold across datasets

Earlier sweeps ([report 11 — cross-dataset synthesis](reports/11_synthesis_cross-dataset.md))
found the same shape on SWE-bench Pro and WebDev: a reasoning planner plus a
sub-cent executor plus a targeted repair turn matches or beats frontier single
models at a fraction of the cost.

Three caveats worth stating plainly:

- **Containment is not free accuracy.** Replacing a paid triage model with the
  harness's digest removes 100% of triage spend ($0.0000 vs ~$0.0018/repair).
  It does not by itself raise pass rates; it lowers the cost of reaching them.
- **Containment does nothing when there is nothing to contain.** A failure whose
  whole output is a handful of lines shows a delta at or below zero, and the
  reports print that as readily as a win.
- **Report 12's containment receipt for the `sj_*` rows is blank** — those arms
  bypassed the instrumentation (fixed; see
  [§4.5](docs/straitjacket-implementation.md#45-refusal-over-fabrication)).
  Pass rates and costs in that report are valid; the residency columns need a
  re-run.

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

**Use `--no-cache` whenever you care about the containment receipt.** Cached
task records from older revisions have no containment field, and a cached run
silently produces an empty receipt.

Only run one sweep at a time. Concurrent runs write the same results file,
cache and reports, and the last one to finish wins.

### Other datasets

```bash
python3 run_benchmark.py --dataset swebench --group straitjacket --n 30 --report
python3 run_benchmark.py --dataset webdev   --group single       --n 10 --report
```

### CLI reference

| Flag | Values | Meaning |
|---|---|---|
| `--dataset` / `-d` | `bcb`, `swebench`, `webdev` | which benchmark |
| `--n` | integer | number of tasks |
| `--variants` / `-v` | comma-separated ids | exactly which architectures to run |
| `--group` / `-g` | `all`, `single`, `combo`, `straitjacket`, `nextgen`, `ablation` | a preset family instead of explicit ids |
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
│   ├── evaluator.py                 #   sandboxed execution + the Evidence contract
│   ├── architectures.py             #   variant registry + every pipeline
│   └── reporter.py                  #   markdown report + HTML dashboard
│
├── docs/                            # ← explanations (methodology, not results)
│   ├── straitjacket-implementation.md
│   ├── pipeline-architecture.md
│   ├── bigcodebench-hard-sweetspot-methodology.md
│   └── webdev-sweetspot-methodology.md
│
├── reports/                         # ← results only, indexed by run order
│   ├── README.md                    #   the index: read this first
│   └── NN_<dataset>_<tag>_n<N>.{md,html}
│
├── tests/                           # contract tests for the straitjacket bridge
├── tools/                           # report indexing, auditing, pricing refresh
│
├── bigCodeBench-hard/               # dataset 1: Python function completion
├── swebench_pro/                    # dataset 2: enterprise git patch resolution
├── webdev/                          # dataset 3: web & networking tasks
│     each: data/  results/  bench_runner.py  + historical sweep scripts
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
