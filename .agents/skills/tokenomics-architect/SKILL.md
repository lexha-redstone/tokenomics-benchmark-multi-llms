---
name: tokenomics-architect
description: Unified multi-LLM decision framework & task-coordination orchestrator. Combines empirical tokenomics ($/solved metric, thinking levels, prompt cache warming, zero-cost triage) with Straitjacket DAG routing (ctx.route/v1, topological parallel waves, CAS checkpoint handoff, and single-tier failure escalation).
---

# Unified Multi-LLM Tokenomics & Orchestration Architect

This skill provides an empirical, production-ready framework for **Multi-LLM Architecture Selection, Tokenomics Optimization, and Cross-Harness Task Orchestration**.

Instead of brute-forcing coding tasks with a single monolithic frontier model ($75+/1M tokens) or running uncontrolled open-loop API calls, this system decomposes complex work into a **Directed Acyclic Graph (DAG)**, routes each subtask by **`Capability × Price × Thinking Level`**, and coordinates parallel execution using **bounded Content-Addressable Storage (CAS) checkpoints**.

---

## 1. The Core Architectural Invariants

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Receipts Before Doctrine ($/Solved Metric):                              │
│    Evaluate architectures strictly by Cost per Solved Task ($/solved).      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Quality Allocation ("Flagship Plans, Cheap Model Implements"):           │
│    Spend Frontier Flagships (Opus/Sol) exclusively on high-leverage         │
│    planning (<250 words); execute bulk code on Standard/Economy models.      │
│    => Matches 98% solo-Opus pass rates at 1.43x–4.2x lower total cost.      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Capability Gating & Price Tie-Breaks:                                    │
│    min_tier (economy < standard < frontier) is a hard exclusion gate.       │
│    Eligible models are ranked by role coverage, then cheapest token price.  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Zero-Cost Triage & Bounded CAS Checkpointing:                            │
│    Never feed raw stack traces or thousands of lines into LLM prompts.      │
│    Extract failure coordinates deterministically ($0.00) and hand off       │
│    compact checkpoint:<id> digests. Resolve slices on-demand via ctx get.   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. Prompt Cache Prefix Warming:                                             │
│    Strip ephemeral noise (timestamps, temp paths, PIDs) to preserve         │
│    96–98% prompt cache hit rates, slashing multi-turn token costs by 10x.   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Declarative Host, Model & Thinking Registry

```mermaid
flowchart TD
    subgraph Tiers ["3-Tier Capability Hierarchy"]
        F[Frontier Tier: plan, reason, architect, deadlock breaker]
        S[Standard Tier: implement, code, edit, reasoning repair]
        E[Economy Tier: explore, search, triage, verify, bulk edit]
    end

    subgraph Thinking ["Thinking Budget Levels"]
        T0[OFF / None: 0 tokens]
        T1[Minimal: ~1k-2k tokens]
        T2[Low: ~2k-4k tokens - Sweet Spot]
        T3[Medium: ~4k-8k tokens - Escalation]
        T4[High / Adaptive: ~8k-16k tokens]
    end

    F --- T4 & T3
    S --- T2 & T1 & T0
    E --- T0
```

| Model Identifier | Provider | Capability Tier | Input / Output ($/1M) | Cache Read ($/1M) | Optimal Role & Thinking Level | Primary Use Case |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| **`gemini-3.5-flash-lite`** | Google | **Economy** | **$0.30** / **$2.50** | **$0.030** | Bulk Executor / Initial Drafter (`thinking: off`) | High-volume edits, exploration behind containment. |
| **`gemini-3.6-flash`** | Google | **Standard** | **$1.50** / **$7.50** | **$0.150** | Workhorse Implementer (`thinking: low`) | Algorithmic design, reasoning repair, clean diffs. |
| **`claude-haiku-4.5`** | Anthropic | **Economy** | **$1.00** / **$5.00** | **$0.100** | Coordinator / Explorer (`thinking: off`) | Fast DAG decomposition and test verification. |
| **`claude-sonnet-4.6`** | Anthropic | **Standard** | **$3.00** / **$15.00** | **$0.300** | Contract Advisor / Reviewer (`thinking: off`) | Formal specification contracts (<250 words). |
| **`claude-opus-4.8`** | Anthropic | **Frontier** | **$15.00** / **$75.00** | **$1.500** | Flagship Planner (`prefer: strong`, `adaptive`) | Architecture, complex multi-file regressions. |
| **`gpt-5.6-sol`** | OpenAI | **Frontier** | **$10.00** / **$40.00** | **$1.000** | Frontier Reasoner (`prefer: strong`, `low`) | Complex mathematical logic and system design. |
| **`gpt-5.6-terra`** | OpenAI | **Standard** | **$2.50** / **$10.00** | **$0.250** | Standard Implementer (`thinking: off`) | Heavy code-gen in Codex environments. |

---

## 3. The 4-Step Orchestration Lifecycle

### Step 1: Task Classification & Decomposition (`ctx.route/v1`)
The cheap coordinator (or deterministic fallback) classifies the task and emits an acyclic DAG:
- **Class A (Algorithmic / Multi-Library)**: Draft (`gemini-3.5-lite`) -> Triage ($0) -> Repair (`gemini-3.7-flash`, `low`) -> **Escalate to `claude-opus-5` when the digest reads `broad`/`stalled`**, not after a fixed number of failures. Measured at BCB-Hard N=148: 81.1% at $0.0353/solved.
- **Class B (Enterprise Repo Bug / multi-file patch)**: Architect Contract (`claude-sonnet`) -> Patch Exec (`gemini-3.5-lite`) -> Triage ($0) -> Targeted Repair (`claude-opus`). **Unvalidated** — this repository has no executed multi-file-patch dataset, so this shape is reasoned from Class A/C, not measured.
- **Class C (Web / Microservices)**: Plan (`gemini-3.7-flash`) -> Exec (`gemini-3.5-lite`) -> Test & Fix (`gemini-3.7-flash`).
- **Class D (High-Throughput Batching)**: Cascade `gemini-3.5-lite` -> `gemini-3.7-flash (min)` -> `gemini-3.7-flash (low)`.
- **Class E (Pure Gemini Native)**: `gemini-3.5-lite` -> `gemini-3.7-flash (low)` -> `gemini-3.7-flash (med)` — 73.0% at $0.0362/solved. **Do not use the `low -> med -> high` thinking ladder**: measured at 75.7% for $0.0674/solved, the worst value of any arm tested.

### Step 2: Capability × Price × Thinking Routing
- Validates graph acyclicity (`_assert_acyclic`) and bounds (`max_nodes <= 12`).
- Evaluates `min_tier`, role coverage, and `thinking_level`.
- Prices the total budget up front against token pricing tables.

### Step 3: Pre-Run Validation & Preview (`--dry-run`)
- Displays wave schedules, model choices, thinking budgets, and estimated dollar costs.
- Rejects plans exceeding user budget limits (`budget_usd`).

### Step 4: Closed-Loop Wave Execution (`run_route`)
1. **Parallel Waves**: Dispatches ready nodes concurrently using `ThreadPoolExecutor`.
2. **CAS Checkpoint Handoff**: Freezes outputs into immutable blobs; passes `checkpoint:<id>` to downstream dependents.
3. **1-Tier Escalation**: Automatically re-runs failed nodes on the cheapest strictly superior model tier.
4. **Bounded Re-planning**: If blocked, coordinator patches the DAG with recovery nodes (capped by `max_replans <= 2`).

---

## 4. Non-Price Dimensions & Empirical Rules

1. **Flood Discipline Invariant**:
   - `gemini-3.5-flash-lite` has low flood discipline (emitted 7.8 MB on uncontained log dumps). **Always run Flash-Lite behind Straitjacket tool wrapping (`ctx run`)**.
   - `gemini-3.6-flash` has high flood discipline (autonomously uses `grep`, emitting <1 KiB).
2. **Context Drag vs. Unit Price**:
   - Accumulating multi-turn context can erase per-token cost advantages. Use CAS checkpointing to keep input prompts bounded.
3. **Measured Throughput**:
   - `gemini-3.6-flash` (91.3 tok/s) is **~36% faster** than `gemini-3.5-flash-lite` (58.8 tok/s).

---

## 5. Production Benchmark Receipts

Read from the three sweeps that are instrumented end to end and reconcile with
their own raw records: **BigCodeBench-Hard N=148**
(`reports/19_bcb-hard_straitjacket_n148.md`, the complete dataset),
**BigCodeBench-Hard N=100** (`reports/12_bcb-hard_straitjacket_n100.md`) and
**ClassEval N=91** (`reports/17_classeval_opus5_n91.md`).

**Rows from different sweeps are not comparable** — attempt budget differs (three
rungs at N=148, two at N=100), which alone moves `claude-opus-5` from 76% to
84.5%. Compare within a sweep only.

| Architecture Pattern | Sweep | Pipeline Configuration | Pass Rate | Cost per Solved Task | Read against the frontier |
|:---|:---|:---|:---:|:---:|:---|
| **`Evidence-gated escalation`** ⭐ | BCB-Hard N=148 | Lite -> `3.7-flash (low)` -> `3.7-flash (med)` -> `opus-5` **when the digest reads `broad`/`stalled`** | **81.1%** | **$0.0353** | 96% of Opus's pass rate for 74% of its spend. **The recommended default.** |
| **`Ladder then frontier`** | BCB-Hard N=148 | Lite -> `3.7 low` -> `3.7 med` -> `opus-5` after every rung fails | **84.5%** | $0.0452 | ties Opus-solo exactly, at 99% of its $/solved — the ladder bought nothing. |
| **`Single: claude-opus-5`** | BCB-Hard N=148 | `claude-opus-5` x3 | **84.5%** | $0.0456 | the accuracy ceiling on the full dataset. |
| **`Gemini ladder, no frontier`** | BCB-Hard N=148 | Lite -> `3.7 low` -> `3.7 med` | 73.0% | $0.0362 | what you get with no frontier budget at all. |
| **`3.7-flash (medium) x3`** ❌ | BCB-Hard N=148 | thinking turned up instead of the model | 75.0% | $0.0684 | **costs 33% more than opus-5 and solves 14 fewer tasks.** |
| **`Whole-class Cascade`** | ClassEval N=91 | `3.5-lite` -> `3.7-flash (low)` -> `3.7-flash (med)` | **80%** | $0.0371 | best non-frontier arm on ClassEval. |
| **`Per-method, one model`** | ClassEval N=91 | every method to `gemini-3.5-flash-lite` | 73% | **$0.0210** | cheapest arm that still clears 70%. |
| **`Single: claude-opus-5`** | ClassEval N=91 | `claude-opus-5`, whole class | **88%** | $0.0464 | the accuracy ceiling on this slice. |

**Four negative results are load-bearing.**

1. **A high-thinking cheap rung is not cheap.** `gemini-3.7-flash` at `medium`
   emits 6,513 output tokens/task against `claude-opus-5`'s 1,221 — it costs more
   and scores worse. Escalate the *model*, not the *thinking budget*.
2. **An attempt-count ladder in front of a frontier model earns nothing** at a
   three-rung budget: identical pass rate, identical cost. If the frontier model
   is going to be called, call it on evidence and sooner.
3. **Routing sub-tasks by labelled difficulty** did not beat the cascade or the
   one-model control on ClassEval (71% at $0.0317 vs 73% at $0.0210).
4. **A repair turn that de-escalates** rescues 16% of failures against 41% for
   one that escalates (z = +3.55, p = 0.0004).

And one positive mechanism worth stating: the evidence gate wins by escalating
**more often** (45% of tasks vs 29%) but **earlier**, skipping the expensive
`medium`-thinking rung that was going to fail anyway. Do not read "call the
frontier model less" as the lesson; the lesson is "stop paying for rungs that
will not work".

---

## 6. Critical Anti-Patterns & Failure Modes

1. ❌ **Monolithic Frontier Brute-Forcing**: Running full multi-turn coding exclusively on Opus/Sol ($75+/1M tokens).
2. ❌ **Uncontained Stderr / Traceback Dumping**: Appending raw 5,000-line pytest dumps into prompts.
3. ❌ **LLM-Based Error Triage (`triage_error`)**: Paying LLM tokens to summarize errors instead of zero-cost local regex profiling ($0.00).
4. ❌ **Prompt Prefix Cache Busting**: Including timestamps, ephemeral `/tmp/...` paths, or PIDs in prompts.
5. ❌ **Unbounded Re-planning**: Allowing endless repair loops without hard wave/budget caps.

---

## 7. Reference Files in This Unified Skill

- [`references/routing_contract.md`](references/routing_contract.md): Coordinator prompt contract, `ctx.route/v1` schema with thinking level support.
- [`references/pricing_and_capabilities.json`](references/pricing_and_capabilities.json): Master pricing, cache rates, thinking levels, and role matrix.
- [`references/catalog_dimensions.md`](references/catalog_dimensions.md): Measured throughput, flood discipline, and the provenance rule.
- [`examples/orchestrator_engine.py`](examples/orchestrator_engine.py): Executable Python reference implementation of the multi-model engine.
- [`examples/production_recipes.py`](examples/production_recipes.py): Executable benchmark patterns (Smart Repair, Ultra-Sweet Hybrid).
- [`examples/example_routes.json`](examples/example_routes.json): Sample DAG specifications for common engineering workflows.
