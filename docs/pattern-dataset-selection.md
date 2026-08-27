# Choosing a dataset where a non-cascade pattern can win

> **Status**: survey complete; the Tier 1 recommendation is **implemented**.
> ClassEval now runs through the ordinary entry point --
> `python3 run_benchmark.py --dataset classeval --group classeval` -- with arms
> in [`src/classeval.py`](../src/classeval.py), a gold-solution preflight in
> [`tools/classeval_preflight.py`](../tools/classeval_preflight.py), and the H1
> verdict in [`tools/analyze_classeval.py`](../tools/analyze_classeval.py).
> **The full N=91 sweep has now run, and H1 is not supported** -- see
> [§6, the verdict](#6-the-verdict-h1-is-not-supported) and
> [report 17](../reports/17_classeval_opus5_n91.md). **H2 remains untested, and
> [§7](#7-what-must-be-run-next-and-the-constraint-that-decides-it) says what to
> run for it.**
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

| | Property | Why H1/H2 needs it | BCB-Hard | ClassEval |
|---|---|---|---|---|
| **P1** | one task decomposes into ≥2 sub-tasks | otherwise there is nothing to route | ✗ single function | ✅ ~4 methods per class |
| **P2** | sub-tasks differ in difficulty, ideally **labelled** | routing needs a gradient to exploit, and a label makes the result attributable rather than anecdotal | ✗ | ✅ the dataset's own `dependencies` tiers; 71/100 classes span more than one |
| **P3** | **per-sub-task** verification | without it you can measure only the total, and cannot show the routing itself did the work | ✗ one pass/fail | ✅ every method ships its own test class |
| **P4** | retry is expensive, or context exceeds one window | this is the axis where a plan is worth paying for in advance | ✗ ~2K tokens, cheap retry | ✗ ~333-token class, cheap retry |

P3 is the one usually missing, and the one that decides whether an experiment
produces evidence or a vibe. A dataset with P1+P2 but no P3 can only tell you
*that* an arm won; P3 tells you *which sub-task the cheap model actually got
right*, which is the entire claim under test.

**Read the two columns correctly — they are not both failures.** BCB-Hard misses
every property, so it cannot test H1 at all; that is not a defect, because H1 is
not what it was run to answer (it answered the containment and escalation-
direction questions, and answered them well). ClassEval hits P1, P2 and P3
cleanly — it is a *suitable* dataset for H1, it was run, and H1 lost on it. The
one property still missing on both is **P4**, and P4 is the whole of H2. So the
open question is not "is there a dataset that can test difficulty routing" —
that one is answered — but **"does any of this survive when the oracle stops
being free?"**

## 3. What was in the repo before ClassEval, and why none of it sufficed

**WebDev is not an independent dataset.** `load_webdev_problems`
([src/datasets.py:125](../src/datasets.py:125)) filters BigCodeBench-Hard rows by
web/networking library imports. It inherits BCB-Hard's shape exactly, so a
result there is not a second observation — it is the same observation on a
subset. Any cross-dataset claim that leans on BCB-Hard *and* WebDev is leaning
on one dataset twice.

**SWE-bench Pro was removed from the repository, and why is worth recording.**
It had exactly the shape H1 needs — measured over the 100 local tasks before
deletion, a mean 5 files touched per gold patch, 87% of them multi-file — but it
was never executed. `run_swebench_pro_task` did not run the repository's tests.
It checked that the candidate touched one of the gold patch's target files, then
scored it **pass** if the candidate string contained either of the first two `+`
lines of the canonical patch. The local JSONL was missing what a real harness
needs: `FAIL_TO_PASS` and `PASS_TO_PASS` were empty lists and `code_context` was
an empty string on every row.

So every SWE-bench Pro number this repository ever printed was a
**canonical-patch substring score**, not a pass rate — a measure of how closely a
model reproduced the gold patch's literal text. Combined with the then-current
simulator, whose SWE branch built diffs out of `canonical_patch`'s own added
lines, that is how a single-`claude-opus-5` arm came to record **10/10 at 199
average output tokens**.

A proxy that scores like a pass rate and prints beside real pass rates is worse
than no dataset at all, so the dataset, its arms, its results and its three
reports (indices 07–09) were **deleted** rather than annotated. Re-adopting
SWE-bench Pro means running the real containerised harness, not writing another
arm.

> **It was re-adopted on exactly those terms, on 2026-08-25.** Nothing from the
> deleted version was carried forward. The current path drives Docker directly
> and lets upstream decide every verdict — its per-instance image, the
> `git checkout` that restores the graded tests, `run_script.sh`, and
> `parser.py` ([`src/evaluator.py`](../src/evaluator.py)). Setup:
> [swebench-pro-setup.md](swebench-pro-setup.md). First valid result:
> [report 31](../reports/31_swebench-pro_straitjacket_n20.md), N=20, directional
> only. Reports 21 and 23 are earlier and defective for a different reason —
> a two-oracle-call budget made every escalation branch unreachable.

So the honest starting position **at the time this section was written** was:
this repo has one working dataset, and it is the one that favours cascades.
It now has three that executed — BCB-Hard, ClassEval and SWE-bench Pro — and
the third has only a directional result.

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

Re-checked against the live sources in August 2026, because two of the three
entries had moved since the survey was written.

| Dataset | Fit | Cost to adopt |
|---|---|---|
| **[FeatureBench](https://github.com/LiberCoders/FeatureBench)** (ICLR 2026) ✅ | features spanning multiple commits and PRs, traced from unit tests along a dependency graph — P1/P2/P4, and the multi-test structure gives a workable P3. 200 tasks over 24 repositories, 3,825 executable environments | Docker required; the **fast split is 100 instances and needs no GPU**, 57.2 s/instance on gold patches. **The floor-out worry has receded**: the paper's 11% was Opus 4.5 on the full set, while the current lite-split leaderboard has Opus 4.7 at **46.7% resolved / 78.2% pass**, GPT-5.5 at 26.7%. That is a usable dynamic range for a cascade |
| **SWE-bench Pro** (real harness) 📄 | 728 instances, a genuine containerised grader (`swebench.harness.run_evaluation`, three patch-apply strategies). Structurally ideal, as §3 measured before the mock copy was deleted | needs containers, the real `FAIL_TO_PASS`/`PASS_TO_PASS` lists and a fresh dataset pull. **But see §7 — the harness is the easy half.** SWE-bench's validity rests on withholding the grading tests, and every arm in this repository is built to iterate against them |
| **[Commit0](https://github.com/commit-0/commit0)** ⚠️ | 54 libraries built from a spec against interactive unit tests — the strongest P1/P4 on the list | **Published numbers are contaminated.** The git history was not squashed, so an agent could `git log` its way to the removed implementation; when that was patched, MiniMax-M2.5 fell from 50.0% to 18.8%. Any baseline read from a pre-patch leaderboard is unusable, and this repository has no `git log` affordance to exploit it either way — so its arms would be measured against numbers nobody else produced under the same rules |

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

Wiring it up surfaced two things the survey did not predict.

**Some classes cannot be scored, and not for the same reason.** On a provisioned
machine 8 of the 100 fail with their own gold solution: two where gold simply
fails its own tests, one written against NumPy 1.x, one that resolves a
hostname, one flaky random-map task, and three whose `methods_info` omits a
method the class needs. That last group fails *only* for the per-method arms, so
it is not a constant subtracted from every arm but a bias against the exact arms
being compared. `tools/classeval_preflight.py` runs gold first and quarantines
them, leaving 92 scorable classes.

**On a bare machine the count doubles, and that is an environment problem
wearing a benchmark's clothes.** ClassEval's tasks import ten third-party
packages; without them 12 tasks fail with `ModuleNotFoundError` and would be
quarantined as if defective. Two machines would then measure different task sets
and their pass rates would not be comparable. The preflight therefore checks
imports *before* running anything, refuses to proceed while a package is
missing, and prints the `pip install` line -- `classeval/requirements.txt` is
derived from the data rather than hand-listed, and a test fails if the two
drift apart. If H1 does not show up on
ClassEval — 71 of 100 tasks carrying a real difficulty gradient, with method-level
scoring — then difficulty routing is unlikely to be rescued by a harder dataset,
and the cheaper conclusion is that the finding generalises.

H1 did not show up. FeatureBench's fast split is still the follow-up that adds
P4 without demanding a GPU -- for H2 rather than H1 -- and
[§7](#7-what-must-be-run-next-and-the-constraint-that-decides-it) sets out why it
is the only candidate that does not collide with how these arms work.

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

## 6. The verdict: H1 is not supported

The sweep ran. Nine arms, 91 of the 92 scorable classes, 376 scorable methods,
live API —
[report 17](../reports/17_classeval_opus5_n91.md), regenerate the analysis with
`python3 tools/analyze_classeval.py`.

| Arm | Class pass | Method pass | Total $ | $/solved | Integration gap |
|---|---|---|---|---|---|
| `ce_single_opus` (frontier baseline) | **80/91** | 357/376 | $3.7120 | $0.0464 | 0 |
| `ce_cascade` (**the shape to beat**) | 73/91 | 341/376 | $2.7094 | $0.0371 | 0 |
| `ce_single_flash` | 70/91 | 334/376 | $2.1593 | $0.0308 | 0 |
| `ce_plan_exec` | 70/91 | 333/376 | $1.8594 | $0.0266 | 0 |
| `ce_single_sonnet` | 66/91 | 332/376 | $1.9332 | $0.0293 | 0 |
| `ce_route_flat` (**the control**) | 66/91 | 340/376 | $1.3886 | **$0.0210** | 1 |
| `ce_route_by_tier` (**the hypothesis**) | 65/91 | 341/376 | $2.0615 | $0.0317 | 1 |
| `ce_plan_route` | 65/91 | 337/376 | $2.6631 | $0.0410 | 1 |
| `ce_single_lite` | 56/91 | 313/376 | $0.4040 | **$0.0072** | 0 |

**H1 asked for one number and did not get it.** `ce_route_by_tier` reaches 71%
against the cascade's 80% (z = −1.39, p = 0.17 — the gap itself is inside noise
at this N), and it does *not* buy that with a lower cost per solved task at a
matched pass rate: $0.0317 vs $0.0371 is 1.17× cheaper for nine points less
accuracy. H1 predicted a win at matched-or-better accuracy. There is no reading
of this table that delivers one.

**The control is what makes the result interpretable, and it is the harsher
half.** `ce_route_flat` runs the identical per-method loop with every method
sent to the same cheap model. It lands at 66/91 for **$0.0210 per solved task** —
one class ahead of the routed arm *and* 34% cheaper per solved task. So whatever
per-method generation is worth here, difficulty routing subtracts from it: the
routed arm spends $0.67 more in total to end up one class behind the control it
has to beat. Adding a planner on top (`ce_plan_route`) makes it worse again —
same 65/91, $0.0410 per solved.

**Why the routing spend does not convert.** Per-model delivery inside
`ce_route_by_tier`: `gemini-3.5-flash-lite` delivered 206/207 of the methods it
was routed, `gemini-3.7-flash` 129/146, and `claude-sonnet-5` 6/23 at $0.5107.
That last row is not a model failing — sonnet only ever holds a *repair* turn
one rung above flash, so 6/23 is a rescue rate on the residue two cheaper rungs
already failed. The cheap rung is not the bottleneck this experiment assumed it
was. It delivers essentially everything routed to it; the cost is concentrated
in a hard tail that no amount of routing *policy* reaches, because reaching it
requires a better model, not a better assignment.

The per-tier breakdown says the same thing from the other side. On the
`method_dep` tier — the hardest, and the one the routing table sends up to
`gemini-3.7-flash` at medium thinking — the routed arm scores 85/102 against the
cascade's 90/102 and opus's 94/102. Routing a method to a stronger model does
not help when what makes the method hard is its dependence on *the other methods
in the same class*, which a per-method prompt cannot show it.

**The integration gap is the cost of decomposition, and it is real but small.**
All three per-method arms carry exactly one class where every method passed its
own test and the assembled class still failed; the whole-class arms carry none.
One class in 91 is not the reason H1 failed, but it is the failure mode that
only decomposition can have, and it is worth watching on a dataset with wider
classes.

**What this licenses, stated narrowly.** ClassEval was picked because it has
the structure H1 needs and BigCodeBench-Hard lacks: 71 of its 100 classes span
more than one labelled difficulty tier, and every method is scored separately.
On that dataset, with per-method attribution, **assigning sub-tasks to models by
labelled difficulty did not beat escalating the whole task, and did not beat
sending every sub-task to the cheapest model either.** The per-arm gaps are
individually inside binomial noise at N=91, so this is a failure to find the
effect rather than a proof it is zero — but H1 was a directional prediction with
a specific cost signature, and neither the direction nor the signature is there.

Nothing here contradicts the BCB-Hard finding; it extends it. Escalation buys
what routing does not, on both datasets tested, because *fail → escalate* uses
an oracle that has already run and difficulty routing uses a label chosen before
anything ran.

**Where to look next.** ClassEval has no P4 — a 333-token class fits one window
and retry is cheap — so H2 is untouched, and the pattern-level question is only
answered under a cheap oracle. FeatureBench's fast split adds P4 without
demanding a GPU, and remains the honest next step.

## 7. What must be run next, and the constraint that decides it

### 7.1 The one property both datasets are missing

H1 is answered. H2 is not, and it is the more consequential of the two:

> **H2.** As the oracle gets more expensive or more partial, the cascade's
> advantage shrinks, because *fail → escalate* stops being a free routing
> signal. Front-loaded planning is then paying for information the tests can no
> longer hand over for nothing.

Every result this repository has produced sits on the *cheap* side of that
conditional. BCB-Hard runs ~6 unit tests in a sandbox for $0; ClassEval runs a
per-method test class for $0. So the headline finding — **escalate on a test
failure rather than plan in advance** — has only ever been measured where the
test failure is free, instant and exact. That is precisely the regime in which
escalation *should* win. The finding is not wrong; it is untested outside the
conditions that produce it.

P4 is what breaks that regime, and neither dataset has it.

### 7.2 The constraint nobody wrote down: this pipeline is verifier-in-the-loop

Before picking a P4 dataset, one property of the arms has to be stated, because
it disqualifies an obvious candidate.

Every arm here feeds the repair turn a digest of the **same test suite that
produces the final grade** — test names, docstrings and assertion messages
included. Run a failing candidate through `_eval_solution` and the payload
handed to the next model looks like this:

```
FAIL: test_case_3 (__main__.TestCases.test_case_3)
Test with an invalid FTP user by raising an exception on login.
AssertionError: Exception not raised
```

That is deliberate and it is the entire subject of the research — the question
this repository asks is *how failing test output should reach the next model*,
which requires failing test output to reach it. On BCB-Hard and ClassEval it is
also fair: the tests are part of the task specification, every arm gets the same
access and the same attempt budget, so arm-vs-arm comparison is sound.

Two consequences follow, and only the second one constrains dataset choice:

1. **Absolute pass rates here are not comparable to published leaderboard
   numbers** for the same dataset, which are typically single-shot. `76%` for
   `claude-opus-5` on BCB-Hard is a with-test-feedback, up-to-N-repairs number.
   Compare arms to each other, never to a public leaderboard row.
2. **A benchmark whose validity depends on withholding the grading tests cannot
   host this pipeline unmodified.** SWE-bench-family evaluation rests on
   `FAIL_TO_PASS` being hidden from the agent. An escalation ladder that reads
   the visible `FAIL_TO_PASS` as its stopping signal and iterates until those
   tests pass is the exact leakage pattern that voids the grade — and that is
   not a hypothetical, it is a documented failure of public-split runs.

So adopting SWE-bench Pro "for real" is not the harness job §4 makes it sound
like. Standing up containers is the easy half. The hard half is a design
decision: either the cascade sees the grading tests (and the number is not a
SWE-bench number), or it is given a *different* signal to escalate on (and the
thing being measured is no longer the mechanism the other two datasets
measured). **That is a reason to not run it next**, not a reason to never run
it.

### 7.3 The ranked answer

| | Dataset | Why it is here | Run it? |
|---|---|---|---|
| **1** | **FeatureBench, fast split (100 instances)** | The only candidate that adds P4 *without* the oracle-visibility conflict: it is test-driven by construction, so feeding test output to a repair turn is the intended mode rather than leakage. Docker mandatory but only **18 prebuilt images** for the fast split, no GPU, 57.2 s/instance on gold. Frontier range is wide enough to measure against — Opus 4.7 at 46.7% resolved on the lite split, GPT-5.5 at 26.7% — and its **Passed Rate** metric gives a continuous gradient even where cheap rungs resolve ~0. Cost breakdown in [§7.5](#75-what-adopting-featurebench-actually-costs) | **Yes — this is the one that must be run** |
| **2** | **SWE-bench Pro, real harness** | Structurally the closest thing to the production case the repo is advising about, and the arms already exist | Only after §7.2 is decided in writing. Otherwise it produces a number that looks like a SWE-bench score and is not one — which is how the mock version went wrong in the first place |

> **This ranking has been overtaken by what actually happened.** FeatureBench
> (rank 1) ran and could not be ranked; SWE-bench Pro (rank 2) was adopted
> instead and produced the only valid expensive-oracle rows in the repository.
> The reason is the one §7.2 is about: FeatureBench's verdict depends on a test
> tree *this repository* rebuilds from a local `test_patch`, and when that
> rebuild fails it fails for every arm — a missing measurement dressed as a hard
> task. SWE-bench Pro deletes that step rather than working around it. §7.2's
> condition was met by writing the verifier contract down explicitly:
> [swebench-pro-setup.md](swebench-pro-setup.md) states what each attempt is
> given and what decides its verdict, and every report says the arms are
> with-test-feedback rather than leaderboard rows. **Read rank 1 and 2 as
> swapped.**
| **3** | **SWE-Lancer** | Difficulty labelled *in money* ($50–$32,000), plus managerial tasks that evaluate the planner role in isolation — the cleanest available test of the half of H1 that ClassEval could only test indirectly | Worth it if access is straightforward; it is the best P2 on the list |
| **—** | **Commit0** | Was ranked as the strongest P1/P4. Its published numbers are contaminated by an unsquashed git history, and the corrected drop was 31 points on one model | **Deprioritised**, unless run purely self-referentially with no comparison to any leaderboard |
| **—** | **TPS-Bench / DevEval / CoderEval** | Good labels, but TPS-Bench is tool scheduling rather than code generation, and DevEval/CoderEval have weak P3 | Not needed — ClassEval already covers the labelled-difficulty question they would re-ask |

### 7.4 What a FeatureBench run would actually settle

Run the same three pattern families that BCB-Hard and ClassEval already ran —
escalation ladder, plan-and-execute, and a single-model control — and one number
decides H2: **the first-repair rescue rate, split by whether the repair turn
escalated or not**, exactly as `tools/analyze_patterns.py` computes it today.

- If escalation still dominates at 57 s and a container per attempt, the
  finding generalises past its cheap-oracle origins and the recommendation in
  the README stands as written.
- If plan-and-execute closes the gap or overtakes, then H2 holds, the current
  guidance is scoped to cheap-oracle workloads, and the README's routing matrix
  needs a fourth column for retry cost.

Either outcome is worth the run, which is the same standard §5 applied to
ClassEval. Floor-out is no longer the risk it looked like: frontier resolve
rates sit in the 20–47% band on the lite split, and FeatureBench's **Passed
Rate** — the fraction of a task's ~62.7 fail-to-pass points that a patch clears
— keeps the gradient measurable even for a rung that resolves nothing outright.
What it costs to get there is in [§7.5](#75-what-adopting-featurebench-actually-costs).

> **Ran at N=48; neither outcome was observed.** A third failure mode this
> section did not anticipate dominated instead: the candidate diff was rejected
> by `git apply` on 77–94% of tasks per arm, so the repair turn under study
> mostly never happened and no pairwise arm difference reaches p < 0.05. The
> Passed Rate hedge above was the right instinct — but `test_pass_ratio` is
> dropped by the result serialiser and never reached a report, so the sweep has
> only the binary verdict it was supposed to be protected from. Full account and
> the three preconditions for a rerun:
> [featurebench-n48-lessons.md](featurebench-n48-lessons.md).

### 7.5 What adopting FeatureBench actually costs

Checked against the upstream repository and paper, because "Docker required" hides
where the real work is.

**Docker is a hard prerequisite, and there is no documented non-Docker path.**
Every instance ships a Dockerfile that defines its execution environment, and the
paper is explicit that "all subsequent operations [are] executed within this
sandboxed environment". Reproducing 24 repositories' conflicting dependency sets
natively is the exact problem containers were adopted to solve, so working around
it is not advisable even where it is possible.

Three things make that cheaper than it first sounds:

| | |
|---|---|
| **Disk is the real prerequisite** | Upstream documents `fb pull --mode fast` as 18 images, but the dataset's `image_name` values are per *instance* — `libercoders/featurebench-specs_packaging-instance_<hash>`, and the un-suffixed repository 404s. The one measured directly is **10.2 GB compressed**, for one of the smallest libraries in the set. Size it with `tools/featurebench_preflight.py --disk` before pulling |
| **They are prebuilt and published** | `fb pull --mode fast` — no local image builds. Total disk footprint is not stated upstream; measure it with that command before committing |
| **No GPU** | 57.2 s/instance on gold patches |

The tooling is a two-stage CLI, and the second stage has a preflight that this
repository should recognise:

```bash
pip install featurebench          # or: clone + uv sync
fb pull --mode fast               # 18 prebuilt images
fb eval -p gold --split fast      # verify gold passes HERE before spending anything
fb infer --agent <...> --split fast
fb eval -p runs/<timestamp>/output.jsonl --split fast
```

`fb eval -p gold` is the direct analogue of `tools/classeval_preflight.py`: run
gold first, find out which instances this machine cannot score, and quarantine
them before they are charged to the models. The ClassEval experience — 8 of 100
classes unscorable on a provisioned machine, and *doubling* on a bare one — says
to run it first, not after a failed sweep.

**The real cost is not installing Docker; it is where the tests run.**
`run_bigcodebench` ([src/evaluator.py](../src/evaluator.py)) executes a candidate
with `sys.executable` in a local sandbox directory. There is no container
abstraction anywhere in the evaluator, and the straitjacket bridge captures from
that host subprocess. FeatureBench's own two-stage design assumes the agent works
*inside* the container during `fb infer` and hands out a `patch.diff` for
independent grading.

That mismatch matters because of what these arms are. An escalation ladder needs
the test result **on every repair turn**, not once at the end — *fail → escalate*
is the routing signal, so the oracle sits in the hot path. Adopting FeatureBench
therefore means giving `src/evaluator.py` a container-backed execution path and
letting `src/straitjacket.py` capture from inside the container rather than from
a host subprocess. That is the work item; `docker pull` is not.

**One upstream metric removes the floor-out risk entirely.** Alongside Resolved
Rate, FeatureBench reports **Passed Rate** — the average fraction of
fail-to-pass tests passed per task, with ~62.7 fail-to-pass points per instance.
So even a `gemini-3.5-flash-lite` rung that resolves close to zero still produces
a continuous gradient to measure escalation against, which is exactly what the
binary pass/fail of BCB-Hard and ClassEval could not give. This is a better fit
for the escalation question than the earlier "cheap arms may floor out" note
assumed.

**Platform note.** These images are published for the usual x86-64 CI targets;
on Apple Silicon expect emulation unless multi-arch variants exist, which is
worth confirming with a single `fb pull` before planning a sweep. A Linux host —
or a remote Docker context pointed at one — avoids the question.

## Sources

- [ClassEval — arXiv 2308.01861](https://arxiv.org/abs/2308.01861) · [GitHub](https://github.com/FudanSELab/ClassEval) · [HuggingFace](https://huggingface.co/datasets/FudanSELab/ClassEval)
- [Commit0: Library Generation from Scratch — arXiv 2412.01769](https://arxiv.org/abs/2412.01769) · [GitHub](https://github.com/commit-0/commit0)
- [FeatureBench — arXiv 2602.10975](https://arxiv.org/abs/2602.10975) · [GitHub](https://github.com/LiberCoders/FeatureBench)
- [SWE-Lancer — arXiv 2502.12115](https://arxiv.org/abs/2502.12115) · [OpenAI](https://openai.com/index/swe-lancer/)
- [TPS-Bench — arXiv 2511.01527](https://arxiv.org/html/2511.01527)
- [DevEval — arXiv 2405.19856](https://arxiv.org/abs/2405.19856)
- [RouterBench — arXiv 2403.12031](https://www.alphaxiv.org/abs/2403.12031)
- [SWE-bench evaluation guide (containerised harness, FAIL_TO_PASS / PASS_TO_PASS)](https://www.swebench.com/SWE-bench/guides/evaluation/)
- [FeatureBench dataset card](https://huggingface.co/datasets/LiberCoders/FeatureBench) · [OpenReview](https://openreview.net/forum?id=41xrZ3uGuI) · [paper HTML](https://arxiv.org/html/2602.10975v1)
