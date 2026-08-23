# Choosing a dataset where a non-cascade pattern can win

> **Status**: survey complete; the Tier 1 recommendation is **implemented**.
> ClassEval now runs through the ordinary entry point --
> `python3 run_benchmark.py --dataset classeval --group classeval` -- with arms
> in [`src/classeval.py`](../src/classeval.py), a gold-solution preflight in
> [`tools/classeval_preflight.py`](../tools/classeval_preflight.py), and the H1
> verdict in [`tools/analyze_classeval.py`](../tools/analyze_classeval.py). No
> full sweep has been run yet; only a 2-task smoke test.
> **Companion**: [README §1 — Why the cascade shape suits this dataset](../README.md#why-the-cascade-shape-suits-this-dataset)

## 1. The hypothesis, stated so it can fail

The N=100 BigCodeBench-Hard result did not say cascades are better. It said
cascades suit *that dataset*, for two measured reasons: there was nothing to
decompose (median gold solution 162 tokens, one function body), and an exact
oracle was free on every turn (median 5 unit tests, $0 to run).

Read as a conditional, that licenses a prediction:

> **H1.** When one task contains several sub-tasks of *unequal* difficulty, a
> pattern that assigns sub-tasks to models by difficulty — planner/executor, or
> a read/write split — should beat a cascade at equal spend, because the cascade
> can only escalate the *whole* task and therefore pays frontier prices on the
> easy sub-tasks it re-solves along the way.

H1 is falsifiable in one number: **cost per solved task at matched pass rate.**
If a difficulty-routed arm cannot beat a cascade there, H1 is dead regardless of
how appealing the architecture is.

A second prediction follows from the oracle property:

> **H2.** As the oracle gets more expensive or more partial, the cascade's
> advantage shrinks, because *fail → escalate* stops being a free routing
> signal. Front-loaded planning is then paying for information the tests can no
> longer hand over for nothing.

## 2. What a dataset needs to test this

| | Property | Why H1/H2 needs it | BCB-Hard |
|---|---|---|---|
| **P1** | one task decomposes into ≥2 sub-tasks | otherwise there is nothing to route | ✗ single function |
| **P2** | sub-tasks differ in difficulty, ideally **labelled** | routing needs a gradient to exploit, and a label makes the result attributable rather than anecdotal | ✗ |
| **P3** | **per-sub-task** verification | without it you can measure only the total, and cannot show the routing itself did the work | ✗ one pass/fail |
| **P4** | retry is expensive, or context exceeds one window | this is the axis where a plan is worth paying for in advance | ✗ ~2K tokens, cheap retry |

P3 is the one usually missing, and the one that decides whether an experiment
produces evidence or a vibe. A dataset with P1+P2 but no P3 can only tell you
*that* an arm won; P3 tells you *which sub-task the cheap model actually got
right*, which is the entire claim under test.

## 3. Neither dataset in this repo can test it

**WebDev is not an independent dataset.** `load_webdev_problems`
([src/datasets.py:125](../src/datasets.py:125)) filters BigCodeBench-Hard rows by
web/networking library imports. It inherits BCB-Hard's shape exactly, so a
result there is not a second observation — it is the same observation on a
subset. Any cross-dataset claim that leans on BCB-Hard *and* WebDev is leaning
on one dataset twice.

**SWE-bench Pro in this repo is not executed.** It has the right shape —
measured over the local 100 tasks:

```
problem statement   mean  351 tok    gold patch   mean 2566 tok (median 1707, max 14301)
files touched       mean    5        multi-file share 87%
```

A 5-file patch is P1 and almost certainly P2. But `run_swebench_pro_task`
([src/evaluator.py:334](../src/evaluator.py:334)) does not run the repository's
tests. It checks that the candidate touched one of the gold patch's target
files, and then scores it **pass** if the candidate string contains either of
the first two `+` lines of the canonical patch
([src/evaluator.py:361](../src/evaluator.py:361)). The local JSONL is missing
what a real harness would need: `FAIL_TO_PASS` and `PASS_TO_PASS` are empty
lists and `code_context` is an empty string on every row.

Two consequences, both worth stating plainly:

1. Every SWE-bench Pro number in this repository is a **canonical-patch
   substring score**, not a pass rate. It rewards reproducing the gold patch's
   literal text. Combined with the pre-fix simulator — whose SWE branch emitted
   diffs built from `canonical_patch`'s own added lines
   ([src/client.py](../src/client.py), `_fallback_dispatch`) — this is how
   `swebench_pro/results/single_opus5_results.json` came to record **10/10 at
   199 average output tokens**.
2. The one dataset here with the structure H1 needs cannot currently answer H1.
   Fixing that means running the real containerised harness, not writing another
   arm.

So the honest starting position is: **this repo has one working dataset, and it
is the one that favours cascades.**

## 4. Shortlist

Ranked by fit × cost to wire in. Provenance is marked because it matters: ✅ =
inspected first-hand from the live dataset, 📄 = from the paper/repo docs.

### Tier 1 — drop-in, and purpose-built for H1

**[ClassEval](https://huggingface.co/datasets/FudanSELab/ClassEval)** ✅ —
100 class-generation tasks, verified by fetching all 100 rows from the same
HuggingFace `datasets-server` endpoint `ensure_bcb_dataset` already uses:

```
methods per class     mean 4.1   median 4   min 2   max 10   (410 methods total)
skeleton              mean  563 tok        solution  mean 333 tok   tests mean 1710 tok
per-method difficulty tiers, from the dataset's own `dependencies` annotation:
    field-dep       172 (42%)      method-dep (hardest)  106 (26%)
    lib-dep          37  (9%)      field+lib              35  (9%)
    standalone (easiest)  60 (15%)
classes spanning >1 difficulty tier                              71 / 100
classes holding BOTH a standalone and a method-dependent method  20 / 100
```

This hits all of P1, P2 and P3 at once, and it is the only candidate that hits
P3 cleanly: every entry in `methods_info` carries its own `test_code`,
`solution_code` and `dependencies`, so you can score **each sub-task separately
and attribute it to the model that wrote it**. `ClassEval_0` is the shape in
miniature — `is_start_with` is `Standalone: true`, `get_jwt_user` pulls in
`datetime`, and `filter` depends on `is_start_with`, so a correct plan must also
get the *order* right.

It does not have P4: a 333-token class still fits one context window and retry
stays cheap. That is a feature for a first experiment — it isolates
decomposition from long-horizon cost — but it means ClassEval can refute H1
without touching H2.

Practical: pure Python with no container, so `run_bigcodebench`
([src/evaluator.py:192](../src/evaluator.py:192)) needs only a per-method
variant. Code MIT, **data CC BY-NC 4.0** — non-commercial, fine for this
repository, worth knowing before anything is republished. The paper already
compares holistic vs incremental vs compositional generation, which is the
monolith-vs-planner axis, so there is a published baseline to sit beside.

### Tier 2 — real P4, real infrastructure cost

| Dataset | Fit | Cost to adopt |
|---|---|---|
| **[FeatureBench](https://github.com/LiberCoders/FeatureBench)** (ICLR 2026) 📄 | features spanning multiple commits and PRs, derived by tracing unit tests along a dependency graph — P1/P2/P4, and the multi-test structure suggests P3 | Docker required; **fast split of 100 instances needs no GPU**, ~57 s/instance on gold patches. Claude 4.5 Opus resolves 11%, so headroom is enormous and cheap arms may floor out |
| **SWE-bench Pro** (real harness) 📄 | already measured above as structurally ideal; the data is already on disk | needs containers + the `FAIL_TO_PASS`/`PASS_TO_PASS` lists this copy lacks. Highest value per unit of work *because the arms already exist* |
| **[Commit0](https://github.com/commit-0/commit0)** 📄 | 54 libraries built from a spec against interactive unit tests — the strongest P1/P4 on the list | SOTA reaches 6–29% pass; a `gemini-3.5-flash-lite` rung will likely score ~0, which compresses the very differences being measured |

### Tier 3 — labelled difficulty, or a different domain

- **[SWE-Lancer](https://arxiv.org/abs/2502.12115)** 📄 — 1,400+ Upwork tasks
  with **payouts from $50 to $32,000**, i.e. a difficulty label denominated in
  money, graded by triple-verified end-to-end tests. It also contains
  *managerial* tasks where the model picks between implementation proposals,
  which is a direct evaluation of the planner role in isolation.
- **[TPS-Bench](https://arxiv.org/html/2511.01527)** 📄 — 200 compounding tasks
  at two difficulty levels with explicit sub-task dependencies and
  parallelisation potential. Purpose-built for H1, but it is tool
  planning/scheduling, not code generation, so it tests the pattern rather than
  this repo's pipeline.
- **[DevEval](https://arxiv.org/abs/2405.19856)** / **CoderEval** 📄 — stratified
  by dependency level (intra-class / intra-file / cross-file; CoderEval uses six
  context levels). Good P2 labels, weaker P3.
- **TaskWeaver** 📄 — a rule-based generator with adjustable difficulty and
  horizon length. The fallback if nothing off-the-shelf isolates the variable:
  synthesise a difficulty gradient instead of hunting for one.

## 5. Recommendation

**Run ClassEval first** -- now wired up, see the README's
[ClassEval section](../README.md#classeval--the-sub-task-routing-experiment).
It was the cheapest to wire in, it is the only candidate with clean per-sub-task
attribution, and its labelled dependency tiers mean a win can be explained
rather than just observed.

Wiring it up surfaced one thing the survey did not predict: **9 of the 100
classes cannot be scored on a clean machine**, and two of those fail *only* for
the per-method arms, because `methods_info` omits a method the class needs. That
is not a constant subtracted from every arm -- it is a bias against the exact
arms the experiment compares. `tools/classeval_preflight.py` runs gold first and
quarantines them, which leaves 91 scorable tasks. If H1 does not show up on
ClassEval — 71 of 100 tasks carrying a real difficulty gradient, with method-level
scoring — then difficulty routing is unlikely to be rescued by a harder dataset,
and the cheaper conclusion is that the finding generalises.

If H1 *does* show up, FeatureBench's fast split is the follow-up that adds P4
without demanding a GPU.

### What to log, or the run proves nothing

The trap is an arm that wins because it spent more, and gets read as a win for
the pattern. Four fields, per sub-task, close it:

| Field | Why |
|---|---|
| `subtask_id`, `difficulty_tier` | the dataset's own `dependencies` label, not one we infer |
| `model_id` that produced it | the routing decision, recorded rather than reconstructed |
| `passed` at method level | P3 — credit attributable to a model and a tier |
| `as_run_usd` per sub-task | so the comparison can be made at *matched cost*, which is the only comparison H1 makes a claim about |

And the arms must be matched on attempts, not just on models. The BCB-Hard
analysis had to fall back to comparing first-repair turns because the cascade
arms got three attempts to `sj_hybrid`'s two
([tools/analyze_patterns.py](../tools/analyze_patterns.py)); designing that
confound out from the start is cheaper than controlling for it afterwards.

**What would falsify H1 on ClassEval:** a difficulty-routed arm that reaches the
same pass rate as the cascade at the same or higher cost per solved task — or
one whose per-tier breakdown shows the cheap model failing on the very
`standalone` methods it was routed for. Either result is worth the run.

## Sources

- [ClassEval — arXiv 2308.01861](https://arxiv.org/abs/2308.01861) · [GitHub](https://github.com/FudanSELab/ClassEval) · [HuggingFace](https://huggingface.co/datasets/FudanSELab/ClassEval)
- [Commit0: Library Generation from Scratch — arXiv 2412.01769](https://arxiv.org/abs/2412.01769) · [GitHub](https://github.com/commit-0/commit0)
- [FeatureBench — arXiv 2602.10975](https://arxiv.org/abs/2602.10975) · [GitHub](https://github.com/LiberCoders/FeatureBench)
- [SWE-Lancer — arXiv 2502.12115](https://arxiv.org/abs/2502.12115) · [OpenAI](https://openai.com/index/swe-lancer/)
- [TPS-Bench — arXiv 2511.01527](https://arxiv.org/html/2511.01527)
- [DevEval — arXiv 2405.19856](https://arxiv.org/abs/2405.19856)
- [RouterBench — arXiv 2403.12031](https://www.alphaxiv.org/abs/2403.12031)
