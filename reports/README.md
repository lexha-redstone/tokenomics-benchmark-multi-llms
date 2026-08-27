# Benchmark Reports

Every sweep this repository has run, in execution order. The number is the
order, not a version: reports are append-only, so a later run never
overwrites an earlier one's evidence.

Regenerate this index with `python3 tools/index_reports.py --apply`.

| # | Date | Dataset | Tasks | Report | Dashboard | What it was |
|---|---|---|---|---|---|---|
| 01 | 2026-07-09 | BigCodeBench-Hard | N=30 | [md](01_bcb-hard_sweetspot_n30.md) | [html](01_bcb-hard_sweetspot_n30.html) | First sweet-spot sweep. gemini-3.1/3.5 + claude-opus-4, pre-straitjacket. |
| 02 | 2026-07-09 | BigCodeBench-Hard | N=30 | — | [html](02_bcb-hard_sweetspot_n30_rev.html) | Revision of the 01 dashboard (opus-4 arm dropped). |
| 03 | 2026-07-13 | WebDev | - | — | [html](03_webdev_sweetspot.html) | WebDev sweet-spot dashboard, pre-straitjacket. |
| 04 | inferred | BigCodeBench-Hard | N=10 | [md](04_bcb-hard_straitjacket_n10.md) | — | First straitjacket pilot. |
| 05 | inferred | BigCodeBench-Hard | N=30 | [md](05_bcb-hard_straitjacket_n30.md) | [html](05_bcb-hard_straitjacket_n30.html) | Straitjacket comparative TCO across all arms. |
| 06 | inferred | BigCodeBench-Hard | N=50 | [md](06_bcb-hard_gemini-vs-claude_n50.md) | — | Gemini vs Claude head-to-head. |
| 10 | inferred | WebDev | N=2 | [md](10_webdev_straitjacket_n2.md) | [html](10_webdev_straitjacket_n2.html) | Straitjacket smoke run on web/networking tasks. |
| 11 | 2026-08-06 | cross-dataset | N=10/30/50 | [md](11_synthesis_cross-dataset.md) | — | Synthesis of every sweep up to that date. |
| 12 | 2026-08-22 | BigCodeBench-Hard | N=100 | [md](12_bcb-hard_straitjacket_n100.md) | [html](12_bcb-hard_straitjacket_n100.html) | Largest sweep. gemini-3.7 + claude-opus-5, full containment receipt. |
| 13 | 2026-08-22 | BigCodeBench-Hard | N=1 | [md](13_bcb-hard_straitjacket_n1.md) | [html](13_bcb-hard_straitjacket_n1.html) | Routing-study smoke run: the ten R1-R10 ladders, one task. |
| 15 | 2026-08-23 | ClassEval | N=2 | [md](15_classeval_straitjacket_n2.md) | [html](15_classeval_straitjacket_n2.html) | ClassEval smoke run: the hypothesis arm and the shape it has to beat. |
| 16 | 2026-08-24 | ClassEval | N=91 | [md](16_classeval_straitjacket_n91.md) | [html](16_classeval_straitjacket_n91.html) | Full sub-task routing comparison, eight arms over the scorable classes. |
| 17 | 2026-08-24 | ClassEval | N=91 | [md](17_classeval_opus5_n91.md) | [html](17_classeval_opus5_n91.html) | Same sweep with the claude-opus-5 baseline merged in. |
| 19 | 2026-08-24 | BigCodeBench-Hard | N=148 | [md](19_bcb-hard_straitjacket_n148.md) | [html](19_bcb-hard_straitjacket_n148.html) | The routing study, run over the COMPLETE dataset. Eleven arms: gemini-3.7 ladders with claude-opus-5 gated behind them. |
| 20 | 2026-08-25 | FeatureBench | N=48 | [md](20_featurebench_straitjacket_n48.md) | [html](20_featurebench_straitjacket_n48.html) | The expensive-oracle study, arms F0a-F3. **Do not rank these rows.** F0a/F0b/F1/F2 were replayed from a cache written at 3 oracle calls while F3 ran live at 2, the labels were rewritten for the newer budget (F1 actually used claude-opus-5 on 41/48 tasks), and F2 carries `routing.degraded` on 45/48. Audit: [docs/featurebench-n48-lessons.md](../docs/featurebench-n48-lessons.md). |
| 21 | 2026-08-25 | SWE-bench Pro | N=50 | [md](21_swebench-pro_straitjacket_n50.md) | [html](21_swebench-pro_straitjacket_n50.html) | First live sweep on the real containerised harness, arms S0a-S3. **Do not rank these rows.** It ran at `SBP_MAX_ORACLE_CALLS = 2`, and `_ladder` evaluates its gate once per repair turn -- so at a 2-call budget against a 2-entry ladder no gate can answer 'escalate' and the frontier rung is unreachable for every arm that names one (the same defect report 22 has). The arms also cover different task counts (40, 47, 49, 50), so they were not scored over one task set. Superseded by 29-31, which run at 3 calls with `routing.frontier_is_reachable` asserted over the registry. |
| 22 | 2026-08-25 | FeatureBench | N=48 | [md](22_featurebench_straitjacket_n48.md) | [html](22_featurebench_straitjacket_n48.html) | Three follow-up arms (F4-F6) aimed at the patch-application failure. At `MAX_ORACLE_CALLS = 2` none of them can reach the frontier rung, so F4 and F5 ran the IDENTICAL ladder and its section 3 'cheapest per solved' claim is a coin flip (Fisher p = 0.27). Audit: [docs/featurebench-n48-lessons.md](../docs/featurebench-n48-lessons.md). |
| 23 | 2026-08-25 | SWE-bench Pro | N=50 | [md](23_swe-bench-pro-candidates_straitjacket_n50.md) | [html](23_swe-bench-pro-candidates_straitjacket_n50.html) | Three candidate architectures (S4 grounded contract, S5 patch-health router, S6 sonnet/opus sweetspot) against the same split. All three resolved 0. **Do not rank these rows** -- same 2-oracle-call defect as 21, same unequal denominators (37, 44, 50). |
| 24 | 2026-08-25 | FeatureBench | N=2 | [md](24_featurebench_straitjacket_n2.md) | [html](24_featurebench_straitjacket_n2.html) | Smoke run of the repository-grounded arm (F7) against its blind twin (F1). Neither resolved; the run was to prove the grounding path executes, not to compare arms. |
| 25 | 2026-08-25 | SWE-bench Pro | N=2 | [md](25_swebench-pro_straitjacket_n2.md) | [html](25_swebench-pro_straitjacket_n2.html) | Single-arm smoke (S0b) on the first build with `SBP_MAX_ORACLE_CALLS = 3` and the guard-aware repair roles. |
| 26 | 2026-08-25 | SWE-bench Pro | N=2 | [md](26_swebench-pro_straitjacket_n2.md) | [html](26_swebench-pro_straitjacket_n2.html) | Repeat of 25 after the reporter gained the ungraded-attempt warning; kept because it is the first report that prints it. |
| 27 | 2026-08-26 | SWE-bench Pro | N=2 | [md](27_swebench-pro_straitjacket_n2.md) | [html](27_swebench-pro_straitjacket_n2.html) | Evidence-gate smoke (S2). First SWE-bench Pro record in which the frontier rung was actually invoked. |
| 28 | 2026-08-26 | FeatureBench | N=2 | [md](28_featurebench_straitjacket_n2.md) | [html](28_featurebench_straitjacket_n2.html) | Repeat of 24 after `$/solved` stopped printing `$0.0000` for an arm that solved nothing. |
| 29 | 2026-08-26 | SWE-bench Pro | N=10 | [md](29_swebench-pro_straitjacket_n10.md) | [html](29_swebench-pro_straitjacket_n10.html) | First SWE-bench Pro sweep whose frontier rung is reachable, three arms. S2 evidence gate 5/10, S0b sonnet solo 4/10, S1 cascade 3/10. Ten rows decides nothing; it sized the N=20 run. |
| 30 | 2026-08-26 | SWE-bench Pro | N=20 | [md](30_swebench-pro_straitjacket_n20.md) | [html](30_swebench-pro_straitjacket_n20.html) | The N=20 sweep without the H2 challenger. Superseded by 31, which is the same three arms plus S3 -- read 31 instead. |
| 31 | 2026-08-27 | SWE-bench Pro | N=20 | [md](31_swebench-pro_straitjacket_n20.md) | [html](31_swebench-pro_straitjacket_n20.html) | **The current SWE-bench Pro result.** Four arms, python only, 3 oracle calls each: S2 evidence gate 8/20, S0b sonnet solo 6/20, S1 cascade 6/20, S3 plan-execute 5/20. The ranking matches BCB-Hard's, but nothing here is significant (every pairwise Fisher p >= 0.50) and 42-56% of attempts never reached the test suite. Directional only -- see README section 1. |

**Gaps in the numbering.** Indices are never reused, so a missing number is a report that was withdrawn:

- **07** — SWE-bench Pro N=10 — withdrawn: the SWE-bench Pro path never ran the repository's tests, so its rows were canonical-patch substring scores, not pass rates.
- **08** — SWE-bench Pro N=30 — withdrawn for the same reason as 07.
- **09** — SWE-bench Pro N=30 (live API) — withdrawn for the same reason as 07.
- **14** — never produced a report file.
- **18** — a byte-identical duplicate of 19, written by the same sweep. Removed so the N=148 result has one address.

SWE-bench Pro was later **re-adopted** on the terms 07–09 failed: upstream's own image, restore command, run script and parser decide every verdict ([`src/evaluator.py`](../src/evaluator.py)). Reports 21 onward are that harness. Nothing from 07–09 was carried forward.

`inferred` dates: exact run timestamps were not recorded for those sweeps.
Their order is derived from task count and model generation, which is why it
is labelled rather than stated as fact.

Methodology documents live in [`../docs/`](../docs/), not here — this
directory holds run results only.
