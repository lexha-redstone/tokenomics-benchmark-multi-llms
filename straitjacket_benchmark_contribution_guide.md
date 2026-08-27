# Straitjacket Benchmark Contribution Guide

> **Standard Operating Protocol & Architecture Guide for Contributing to the Multi-LLM Tokenomics Benchmark Suite**

---

## 1. Overview & Core Philosophy

This repository evaluates **Multi-LLM collaboration architectures, cascading strategies, and context containment harnesses** across real-world software engineering benchmarks: [BigCodeBench-Hard](bigCodeBench-hard) and [ClassEval](classeval) (cheap sandbox oracle), [SWE-bench Pro](docs/swebench-pro-setup.md) and [FeatureBench](docs/featurebench-setup.md) (containerised, expensive oracle), plus [WebDev](webdev), which is a library-filtered subset of BigCodeBench-Hard rather than an independent dataset.

The benchmark follows the **Straitjacket Benchmark Charter** and **Tokenomics Initiative** principles:

1. **Receipts Before Doctrine**: Performance and cost claims must be backed by reproducible empirical receipts (JSON run logs + Markdown/HTML reports), not adjectives.
2. **Total Cost of Ownership (TCO) & Cost per Solved Task ($/solved)**: Raw accuracy is meaningless without cost accounting. The primary metric is:
   $$\text{Cost per Solved Task} = \frac{\sum \text{as\_run\_usd}}{\text{Total Solved Tasks}}$$
3. **Context Containment**: In multi-turn repair loops, raw stack traces inflate every later turn's prompt, and LLM-based triage (`triage_error`) buys brevity with a paid round trip. `straitjacket` captures the run at its birth gate and hands the repair turn a bounded, coverage-attested digest from the upstream profile registry (`unittest/v1`, `pytest/v2`, …) at **$0.0000 and no API call** — with `ctx get` / `ctx search` addresses for everything it omitted, so shorter does not mean lost.
4. **Prompt Cache Prefix Warming**: Deterministic stripping of ephemeral paths, timestamps, and ANSI escape codes keeps prompt prefixes identical across attempts, maintaining high prompt cache hit rates.
5. **External Corpora as Teachers, Never Referees**: Benchmark tasks provide real hostile outputs and failure structures to evaluate architectural resilience.

---

## 2. Repository Architecture & Layout

The repository separates **explanations** (`docs/`) from **results** (`reports/`),
with a shared core library (`src/`), analysis utilities (`tools/`), and
dataset-specific benchmark directories:

```
tokenomics-benchmark-multi-llms/
├── README.md                                    # Repository overview, findings and quick start
├── MODELS.md                                    # Supported models, IDs, and pricing table
├── requirements.txt                             # Pinned Python package dependencies
├── straitjacket_benchmark_contribution_guide.md # This guide
│
├── run_benchmark.py                             # 🚀 MASTER UNIFIED CLI RUNNER (every dataset)
├── run_classeval_opus5.py                       # Opt-in single arm: opus-5 on ClassEval, then merge
│
├── src/                                         # Shared Core Benchmark Library
│   ├── config.py                                # Centralized model IDs, pricing, and prompt roles
│   ├── client.py                                # Vertex AI + Anthropic dispatch, retry, usage accounting
│   ├── straitjacket.py                          # The ONLY bridge to ctx-harness (capture, digest, retrieval)
│   ├── routing.py                               # Evidence-gated escalation: classify(), GATES,
│   │                                            #   frontier_is_reachable()
│   ├── evaluator.py                             # Sandbox / Docker execution + the Evidence contract
│   ├── datasets.py                              # Unified dataset loaders (BCB, WebDev, ClassEval,
│   │                                            #   FeatureBench, SWE-bench Pro)
│   ├── architectures.py                         # Variant registry + every BCB/WebDev pipeline
│   ├── classeval.py                             # ClassEval arms: per-method routing + its control
│   ├── featurebench.py                          # FeatureBench arms (expensive oracle; superseded)
│   ├── swebench_pro.py                          # SWE-bench Pro arms (expensive oracle; current)
│   ├── sweep.py                                 # The per-arm run loop, shared by every runner
│   ├── merge.py                                 # Fold a single-arm run back into a sweep's results
│   ├── paths.py                                 # Portable path rendering — no absolute literals
│   └── reporter.py                              # Markdown TCO report & HTML dashboard generator
│
├── docs/                                        # 📚 Methodology & design docs (not results)
│   ├── straitjacket-implementation.md           # The harness bridge, and how it differs from upstream
│   ├── pipeline-architecture.md                 # How one task flows through the system
│   ├── routing-study.md                         # The N=148 design and its result
│   ├── pattern-dataset-selection.md             # Which datasets can falsify which hypothesis
│   ├── swebench-pro-setup.md                    # Docker prerequisites + the per-attempt contract
│   ├── featurebench-setup.md                    # Same, for the superseded dataset
│   ├── featurebench-n48-lessons.md              # Why the N=48 sweep is confounded, and the rules it produced
│   └── *-sweetspot-methodology.md               # Per-dataset experiment design (historical)
│
├── reports/                                     # 📊 RUN RESULTS ONLY, indexed by execution order
│   ├── README.md                                # The index — start here; defective sweeps are labelled
│   └── NN_<dataset>_<tag>_n<N>.{md,html}        # Append-only; a sweep never overwrites an earlier one
│
├── visualization/                               # Charts, each regenerable from the raw result JSON
├── tests/                                       # Contract tests (dispatch policy, routing guards,
│                                                #   straitjacket bridge, portability, reporters)
│
├── tools/                                       # 🛠️ Analysis, preflight, auditing
│   ├── analyze_router_study.py                  # Recomputes the BCB-Hard N=148 table
│   ├── analyze_classeval.py                     # Recomputes ClassEval N=91; refuses a missing control
│   ├── analyze_patterns.py                      # Recomputes the N=100 pattern-family decomposition
│   ├── classeval_preflight.py                   # Deps check + gold run + per-machine quarantine
│   ├── featurebench_preflight.py                # Docker image sizing, pull, gold run
│   ├── swebench_pro_preflight.py                # Split listing, image pull, gold run
│   ├── index_reports.py                         # Assigns report indices & regenerates reports/README.md
│   ├── audit_cache.py / rerun_incomplete.py     # Cache integrity + resume
│   └── generate_n30_report.py / generate_n50_report.py / update_all_reports_pricing.py   # historical
│
├── bigCodeBench-hard/                           # Dataset 1: Python function completion (148 rows, in-repo)
├── classeval/                                   # Dataset 2: class generation, scored per method
│                                                #   data/ also holds quarantine-<split>.json + requirements.txt
├── swebench_pro/                                # Dataset 3: 731 rows, upstream's own grading
│                                                #   data/, run_scripts/, results/ are fetched on demand
│                                                #   and GITIGNORED — they are not source
├── featurebench/                                # Dataset 4: multi-file features (superseded for H2)
├── webdev/                                      # A library-filtered BCB-Hard subset, NOT an independent dataset
│
└── .agents/skills/                              # Bundled agent skills: straitjacket, tokenomics-architect
```

---

## 3. Unified CLI Runner (`run_benchmark.py`)

The primary entry point is `run_benchmark.py`, which allows running any dataset, filtering variants, and generating reports in one shot.

### Basic Usage

```bash
# 1. Run BigCodeBench-Hard on 100 tasks with all Straitjacket variants and auto-generate reports
python3 run_benchmark.py --dataset bcb --group straitjacket --n 100 --report

# 2. Run BigCodeBench-Hard on 10 tasks with specific variants
python3 run_benchmark.py --dataset bcb --variants sj_cascade,sj_hybrid,sj_smart_repair --n 10 --report

# 3. Run WebDev evaluation across single models
python3 run_benchmark.py --dataset webdev --group single --n 10 --report

# 4. Run the ClassEval sub-task routing comparison with fresh execution (no cache)
python3 run_benchmark.py --dataset classeval --group classeval --n 91 --no-cache --report

# 5. Run the SWE-bench Pro expensive-oracle study (needs Docker; prove the harness first)
python3 tools/swebench_pro_preflight.py --gold 3 --write
SBP_LANGUAGES=python python3 run_benchmark.py --dataset swebench-pro --group sbp --n 20 --report --no-cache
```

### Supported CLI Arguments

| Argument | Shorthand | Description | Default |
|---|---|---|---|
| `--dataset` | `-d` | `bcb` (BigCodeBench-Hard), `webdev`, `classeval`, `featurebench`, `swebench-pro` | `bcb` |
| `--group` | `-g` | `all`, `single`, `combo`, `straitjacket`, `nextgen`, `ablation`, `router`, `classeval`, `featurebench`, `fb_grounded`, `sbp`, `sbp_candidates` | `all` |
| `--variants` | `-v` | Comma-separated list of specific variant IDs | `None` (uses group) |
| `--n` | `-n` | Number of tasks to evaluate | `30` |
| `--tasks` | | Run exactly these task ids (comma-separated, or `@file`). Overrides `--n` — useful when only some Docker images are present locally | `None` |
| `--split` | `-s` | Dataset split | Dataset default |
| `--allow-simulation` | | On an unrecoverable API failure, substitute SIMULATED output instead of discarding and retrying. **Off by default** — a 504 or an expired credential must not become a datapoint | `False` |
| `--task-retries` | | How many times to re-attempt a task whose API calls failed; the partial record is discarded each time | `1` |
| `--no-cache` | | Force fresh API execution, ignoring the task cache. **Use whenever an arm definition or a shared constant has moved** — a cached record carries no code version | `False` |
| `--out` | `-o` | Custom output path for JSON result metrics | `<dataset>/results/<dataset>_<group>_results.json` |
| `--report` | `-r` | Generate the Markdown TCO report and HTML dashboard in `reports/` | `False` |

Run `python3 run_benchmark.py --help` for the authoritative list — this table is
maintained by hand and the parser is not.

---

## 4. Step-by-Step Contribution Workflows

### A. Adding a New Model

1. **Register the Model ID in `src/config.py`**:
   ```python
   NEW_MODEL_ID = "gemini-3.7-flash"
   ```
2. **Add Pricing Rates to `PRICING` table in `src/config.py`** (USD per 1M tokens):
   ```python
   PRICING[NEW_MODEL_ID] = {
       "input": 1.50,
       "output": 7.50,
       "cache_read": 0.15,
       "cache_write": 1.50,
   }
   ```
3. **Verify API Routing in `src/client.py`**:
   - Ensure the model prefix (`gemini-*` vs `claude-*`) dispatches to the correct SDK or endpoint.
4. **Update `MODELS.md`** with the new model's specifications.

---

### B. Adding a New Benchmark Architecture / Variant

1. **Implement the Pipeline in `src/architectures.py`**:
   - Structure the function to accept a `problem` dict.
   - Use `dispatch_model(model_id, prompt, thinking_level=..., problem=...)` for LLM calls.
   - Name the evidence treatment rather than hard-coding it: take an
     `error_treatment` parameter and call
     `_treat_error(err, error_treatment, problem=problem, is_swe=is_swe)`.
     It dispatches to `native` (raw tail), `llm` (paid triage model), or
     `straitjacket` (the harness's own digest). The registry's `triage_mode`
     label **must** name the treatment the arm actually applies.
   - Decorate the arm with `@_arm()`, or `@_arm(sj_required=True)` if it claims
     containment — the decorator resets the per-task containment ledger and,
     for straitjacket arms, refuses to start when `ctx-harness` is missing.
   - Never re-summarise, keyword-filter, or tail-truncate a failure yourself.
     `run_bigcodebench` already returns `Evidence` carrying the real digest and
     an addressable handle; producing a second "digest" on top of it is the
     anti-pattern this suite exists to measure against.
   - Return standard metrics dict:
   ```python
   def run_my_custom_pipeline(problem, planner_model=GEMINI_36_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID):
       """Custom multi-model collaboration architecture."""
       # 1. Planning phase
       # 2. Execution phase
       # 3. Verification & Straitjacket repair phase
       return {
           "passed": passed,
           "as_run_usd": round(tot_usd, 6),
           "output_tokens": tot_out,
           "total_tokens": tot_tok,
           "seconds": round(elapsed, 1),
           "error": "" if passed else err,
           "repair_loops": loop_count,
           "triage_usd": round(triage_usd, 6),   # what the treatment actually spent
           "patch": candidate_code_or_patch,
       }
       # `@_arm()` appends "containment": the harness's own accounting of raw
       # tokens captured, digest tokens rendered, and evidence tokens sent.
   ```
2. **Register the Variant in `VARIANT_REGISTRY` in `src/architectures.py`**:
   ```python
   register_variant(
       id="custom_pipeline_sj",
       name="Custom Pipeline (Flash Plan -> Lite Exec + SJ Triage)",
       category="3. Combination + straitjacket",
       dataset_compatibility=["bcb", "webdev", "classeval"],
       fn=run_my_custom_pipeline,
   )
   ```

---

### C. Adding a New Dataset

The mechanical part is three steps; the part that decides whether the dataset is
worth anything is the four rules after them.

1. **Place raw data in `<dataset_name>/data/`** as JSON Lines. If the split is
   large or fetched on demand, gitignore it — `swebench_pro/data/` is the
   precedent. A 24 MB checked-in split is not source.
2. **Implement the loader in `src/datasets.py`**, mapping to the standard task
   fields (`task_id`, `complete_prompt`, `test`, `entry_point`).
3. **Register the evaluation runner in `src/evaluator.py`**, and put the arms in
   their own module (`src/classeval.py`, `src/swebench_pro.py`) rather than
   growing `src/architectures.py`.

**Then, before spending anything:**

4. **Write a preflight that runs the dataset's own gold solutions first**, and
   quarantine what gold cannot pass with a *typed reason* per exclusion
   (`missing_module:PyPDF2` names the package, so it can be fixed rather than
   accepted). A task gold itself fails is charged to the models otherwise.
   **The quarantine file is environment-specific — regenerate it per machine and
   never copy it between them**, or two boxes measure different task sets while
   reporting comparable-looking pass rates.
5. **Never score by proxy.** If the dataset's real verdict is expensive to
   compute, that is a reason to run the real harness or drop the dataset — not a
   reason to approximate it. A similarity score printed in a column labelled
   "pass rate" is worse than no dataset, and this repository deleted one for
   exactly that.
6. **Ship the hypothesis arm's control in the same sweep.** A hypothesis arm
   that beats an unrelated baseline proves nothing; `ce_route_flat` is what
   turned "difficulty routing wins" into "writing method-by-method was doing the
   work". `tools/analyze_classeval.py` refuses to bless a result whose control
   is absent — do the same for a new dataset.
7. **Assert that every arm's distinguishing branch can actually fire.** For a
   ladder, that is `routing.frontier_is_reachable(gate, n_tiers, budget)`
   checked over the registry at import. Three sweeps in this repository shipped
   complete, live, plausible results tables in which the escalation branch was
   unreachable code and nothing said so.

---

## 5. Result Schema & Reporting Contract

### JSON Result Metric Schema (`<dataset>/results/*.json`)

A sweep writes **one file holding every arm**, not one file per arm. The shape,
verbatim from `bigCodeBench-hard/results/bcb_router_results.json`:

```jsonc
{
  "dataset": "bcb",
  "dataset_name": "BigCodeBench-Hard",
  "group": "router",
  "n": 148,

  // Harness provenance for the whole sweep — this is what makes a
  // containment claim auditable after the fact.
  "straitjacket": {
    "backend": "library",          // `library` is required for evidence gating
    "available": true,
    "ctx_version": "0.35.1",
    "raw_cap_chars": 2500,         // what the UNCONTAINED arm was allowed to send
    "frame_budget": { "frame_chars_used": 95, "frame_chars_budget": 160,
                      "frame_fits": true }
  },

  "summary": [                     // one entry per ARM
    {
      "id": "r9_opus_on_evidence",
      "name": "R9. Evidence gate: escalate to opus-5 on broad/stalled",
      "category": "6. Routing study",
      "models": "...",
      "triage_mode": "straitjacket",   // MUST name the treatment actually applied

      "n": 148, "passed": 120, "pass_rate": 0.811,
      "total_as_run_usd": 4.2374, "total_triage_usd": 0.0,
      "cost_per_solved_usd": 0.0353,
      "avg_output_tokens": 1221.0, "seconds": 4821.3,

      // Honesty fields. A non-empty `incomplete_tasks` or a non-zero
      // `simulated_tasks` disqualifies the row from a comparison table.
      "completed": 148, "incomplete_tasks": [], "simulated_tasks": 0,
      "simulation_allowed": false,

      // The containment receipt. `delta_vs_native_tokens` is the A/B
      // (`native_baseline − sent`), NOT `raw − sent`.
      "containment": {
        "captures": 231, "treatment_events": 231,
        "raw_tokens_est": 0, "digest_tokens_est": 0,
        "evidence_sent_tokens_est": 0, "native_baseline_tokens_est": 0,
        "delta_vs_native_tokens": 0, "tokens_kept_out": 0,
        "containment_ratio": 0.0, "profiles": ["unittest/v1"],
        "treatments": ["straitjacket"]
      },

      "results": [                 // one entry per TASK
        {
          "task_id": "BigCodeBench/13",
          "passed": true,
          "as_run_usd": 0.00284, "triage_usd": 0.0,
          "output_tokens": 380, "total_tokens": 2840,
          "repair_loops": 1, "simulated_calls": 0,
          "containment": { /* per-task, same keys as above */ },
          "retrievals": [],
          // The router's own record of what it did. `degraded: true` means the
          // gate wanted typed evidence and did not get it — such a row must
          // never be presented as an evidence-gated result.
          "routing": { "degraded": false,
                       "rungs": ["gemini-3.7-flash/low", "claude-opus-5/off"],
                       "decisions": [], "frontier_used": true,
                       "frontier_rung": "claude-opus-5/off",
                       "solved_at": "claude-opus-5/off" },
          "error": ""
        }
      ]
    }
  ]
}
```

**Three fields exist only so a bad row can be caught later**, and analysis code
should read them before it reads `pass_rate`: `simulated_tasks` /
`simulation_allowed` (was any of this invented?), `incomplete_tasks` (did every
arm cover the same task set?), and `routing.degraded` (did the gate actually
have evidence to gate on?). Dataset-specific runners add their own — SWE-bench
Pro and FeatureBench records carry a guard-failure reason per attempt, because a
pass rate over attempts that never reached the test suite is not a pass rate.

### Auto-Generated Reports in `reports/`

Reports are an append-only log indexed by execution order, so a new sweep never
overwrites an earlier one's evidence. After a run, adopt the new files and
refresh the index with `python3 tools/index_reports.py --apply`.


- **Markdown Report (`reports/NN_<dataset>_<tag>_n<N>.md`)**: Formatted comparative TCO table, $/solved rankings, error breakdown (distinguishing environment/quota errors from algorithmic bugs).
- **Interactive HTML Dashboard (`reports/NN_<dataset>_<tag>_n<N>.html`)**: Mobile-responsive dashboard with KPI scorecards, pass rate bar charts, and detailed architecture specifications.

---

## 6. Pre-Commit Verification Checklist

Before submitting a Pull Request or committing changes:

- [ ] **No Duplicate Logic**: Client calls, pricing definitions, and triage harnesses reside strictly in `src/`.
- [ ] **Refuse Rather Than Fabricate**: An unrecoverable dispatch failure — expired credentials, a 504, an unavailable backend — **raises `DispatchError`**; it does not invent a plausible-looking row. `src/sweep.py` discards the affected record and drops it from the arm's denominator. Simulation is opt-in only (`--allow-simulation` / `ALLOW_SIMULATION=1`) and every simulated call is stamped `usage["simulated"] = True` so it is visible after the fact. Pinned by `tests/test_dispatch_failure_policy.py`. *(This replaced an earlier "fall back gracefully to deterministic simulation" rule, under which a 504 became an ordinary-looking benchmark datapoint.)*
- [ ] **Containment Integrity**: Every digest in a `*_straitjacket` variant comes from the upstream harness via `src/straitjacket.py`. No local re-implementation, no keyword or head/tail selection, no arm that silently degrades when `ctx-harness` is absent.
- [ ] **Truth in Labelling**: A variant's `triage_mode` names the treatment it actually applies (`native` / `llm` / `straitjacket`), and its reported `triage_usd` is what it actually spent.
- [ ] **Fair Baseline**: The uncontained arm gets the failing stream, truncated once by `SJ_RAW_CAP` and nowhere else. Never re-truncate inside an arm, and never hand the native path stdout chatter it would not otherwise have forwarded — degrading the baseline biases every comparison toward straitjacket.
- [ ] **No Re-flooding the Harness**: Nothing on the per-task path may materialise a whole captured stream. Use `ContainedRun.raw_tail(stream, nbytes)`; `raw_stdout` / `raw_stderr` are unbounded and exist for tests and debugging only.
- [ ] **Reproducible Evidence**: Identical failing code produces an identical run handle across processes. If you add a runner or an evaluator, pin whatever it prints that is not evidence (elapsed times, PIDs, temp paths, hash-ordered output) — otherwise every attempt mints a fresh artifact for the same failure.
- [ ] **Self-Consistent Receipt**: An arm whose treatment *is* the baseline must report `delta_vs_native_tokens == 0`. The A/B is `native_baseline − sent`, counted over the same events; `captured − sent` is a larger number and is not the A/B.
- [ ] **Contract Tests Pass**: `pytest tests/test_straitjacket_integration.py -q` is green on the `library` backend.
- [ ] **No Hardcoded Absolute Paths**: All file lookups must use relative paths or `os.path.dirname(os.path.abspath(__file__))`.
- [ ] **Requirements Up to Date**: Any newly introduced package is pinned in `requirements.txt`.
- [ ] **No Machine-Specific Paths in Committed Evidence**: Captured tracebacks carry the absolute paths of whoever ran the sweep. Before committing a results file, rewrite home-directory prefixes to a placeholder. This touches only string fields — re-run `tools/analyze_router_study.py` and `tools/analyze_classeval.py` and diff the output against the pre-redaction run to prove no measured quantity moved.
- [ ] **Clean Git Workspace**: No stray `.DS_Store`, `__pycache__`, or scratch scripts left untracked.
