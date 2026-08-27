# Supported Models & Architecture Pricing Table

Model identifiers, the Vertex AI rates every cost figure in this repository was
computed from, and the architecture families those models are wired into.

**This file is documentation of [`src/config.py`](src/config.py), not a second
source of truth.** If the two disagree, `src/config.py` is right and this file
is stale — every reported dollar came from that table.

---

## 1. Pricing table

USD per **1,000,000 tokens**, mirrored from `src/config.PRICING`.

### 1.1 The four models that produced every headline result

| Model ID | Provider | Tier | Input | Output | Cache Read | Cache Write |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `gemini-3.5-flash-lite` | Google | Economy | $0.30 | $2.50 | $0.030 | $0.00 |
| `gemini-3.7-flash` | Google | Standard — the workhorse | $1.50 | $7.50 | $0.150 | $0.00 |
| `claude-sonnet-5` | Anthropic | Standard | $2.00 | $10.00 | $0.200 | $2.50 |
| `claude-opus-5` | Anthropic | **Frontier** | $5.00 | $25.00 | $0.500 | $6.25 |

**Read the ratio, not the tier label.** The frontier model's output token is
only **3.3×** the standard Gemini's. A cheap rung that emits 5.3× the output
tokens is therefore *already more expensive than the frontier model* — which is
exactly what `gemini-3.7-flash` at `medium` thinking did on the N=148 sweep
(6,513 output tokens/task against Opus-5's 1,221: 33% more money, 14 fewer
solved tasks). A pricing table quoted from memory, with the frontier model at
$15/$75, inverts that decision.

### 1.2 Also priced, used only in historical sweeps

These appear in `PRICING` so that pre-2026-08 reports can be re-costed. No
current sweep calls them.

| Model ID | Provider | Input | Output | Cache Read | Cache Write |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `gemini-3.1-flash-lite` | Google | $0.25 | $1.50 | $0.025 | $0.00 |
| `gemini-3.5-flash` | Google | $1.50 | $9.00 | $0.150 | $0.00 |
| `gemini-3.1-pro-preview` | Google | $2.00 | $12.00 | $0.200 | $0.00 |
| `claude-opus-4-8` | Anthropic | $5.00 | $25.00 | $0.500 | $6.25 |

> `GEMINI_36_FLASH_ID` is an alias for `gemini-3.7-flash`. Older scripts and
> reports referring to "3.6-flash" resolve to 3.7-flash and are priced as such.

> **API dispatch.** Gemini runs on Google Cloud Vertex AI through the
> `google-genai` SDK with adaptive `ThinkingConfig` headroom; Claude runs on the
> Vertex AI Anthropic `rawPredict` endpoint with cache-token accounting. An
> unrecoverable failure on either path **raises** — see the failure policy in
> [`src/client.py`](src/client.py).

---

## 2. Thinking budgets, and what they cost

| Level | Rough budget | When it is right |
| :--- | :--- | :--- |
| `off` / `none` | 0 | Contracts, plans, and any Claude rung here. |
| `low` | ~2k–4k | **The default sweet spot** for test-assertion repair. |
| `medium` | ~4k–8k | **Measured negative on the economy/standard tiers.** Algorithmic deadlocks only. |
| `high` | ~8k–16k | Nothing in these sweeps rewarded it. |

Priced on the full BigCodeBench-Hard dataset: `gemini-3.7-flash` at `low` scored
**71.6% for $4.27**; the same model at `medium` scored **75.0% for $7.60**.
Three and a half points cost 78% more money. **Prefer escalating the *model*
over the *thinking budget*** whenever both are available.

---

## 3. Architecture families

The authoritative list is the variant registry, not this section:

```bash
python3 -c "import sys;sys.path.insert(0,'.');from src.architectures import VARIANT_REGISTRY as R;[print(f'{k:<28}{v[\"name\"]}') for k,v in R.items()]"
```

What the families are, and what the sweeps said about each:

| Family | Shape | Verdict |
| :--- | :--- | :--- |
| **Single model** (`single_*`, `r0*`, `ce_single_*`) | one model writes and repairs | `claude-opus-5` is the accuracy ceiling on both cheap-oracle datasets (84.5% / 88%) and the most expensive per solved task. |
| **Cascade / escalation** (`sj_cascade`, `r6`, `ce_cascade`) | escalate to a stronger model whenever the tests fail | The best non-frontier shape on ClassEval (80%). In front of Opus-5 at a three-rung budget it bought **nothing**. |
| **Evidence-gated escalation** (`r9`, `*_evidence_gate`) | escalate when the harness's typed digest reads `broad`/`stalled` | **The recommended default where retry is cheap** — 96% of frontier accuracy for 74% of frontier spend. |
| **Advisor / executor** (`combo_read_write`, `*_plan_exec`) | a planner writes a contract, a cheap executor implements it | Pays on a genuinely decomposable task (ClassEval, 77% at $0.0266) and loses where there is nothing to decompose. |
| **Sub-task routing** (`ce_route_by_tier`) | assign each sub-task to a tier by its labelled difficulty | **Lost to its own flat control** on both accuracy and cost. |
| **Collaboration / dual-candidate** (`sj_dual_verifier`) | several candidates, then a synthesis turn | Ranked below both escalation shapes; a resample from the same model rescued 0 of 39 failures. |

Full numbers, significance and mechanism: [README §1](README.md#1-key-takeaways--best-setting).

---

## 4. Straitjacket containment: what it does and does not buy

Multi-turn repair loops that paste raw stderr into the next prompt pay three
times: input-token bloat on every later turn, an LLM triage call to summarise
(~$0.0018/repair), and prompt-cache misses from ephemeral paths and timestamps.
The `straitjacket` harness captures the run at its birth gate and hands the
repair turn a bounded, coverage-attested digest from the upstream profile
registry, at **$0.00 and no API call**, with `ctx get` / `ctx search` addresses
for everything omitted.

**What it buys, stated exactly:**

- **Triage spend → $0.0000**, against ~$0.0018 per repair for an LLM triage turn.
- **Stable prompt prefixes**, so identical failures produce byte-identical
  digests (upstream A/B: 96.5–98.1% cache hit rate).
- **Residency**: on BigCodeBench-Hard N=148 the digest was ~52% smaller than
  what the untreated path would have forwarded. On a repository-scale suite
  (SWE-bench Pro) that same delta is **negative** — the digest's failing-test
  census is legitimately larger than a 2,500-character tail. The number is
  meaningless without its dataset and its `SJ_RAW_CAP`.

**What it does not buy:** accuracy. Containment does not by itself raise pass
rates; it lowers the cost of reaching them. Any claim that it "achieves higher
task resolution" is not supported by anything measured here — the arms that
raise pass rates are the ones that escalate to a better model.
