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
| 21 | - | - | - | [md](21_swebench-pro_straitjacket_n50.md) | [html](21_swebench-pro_straitjacket_n50.html) | Comparative TCO Report: `straitjacket` on `SWE-bench Pro` (N=49) |
| 22 | 2026-08-25 | FeatureBench | N=48 | [md](22_featurebench_straitjacket_n48.md) | [html](22_featurebench_straitjacket_n48.html) | Three follow-up arms (F4-F6) aimed at the patch-application failure. At `MAX_ORACLE_CALLS = 2` none of them can reach the frontier rung, so F4 and F5 ran the IDENTICAL ladder and its section 3 'cheapest per solved' claim is a coin flip (Fisher p = 0.27). Audit: [docs/featurebench-n48-lessons.md](../docs/featurebench-n48-lessons.md). |
| 23 | - | - | - | [md](23_swe-bench-pro-candidates_straitjacket_n50.md) | [html](23_swe-bench-pro-candidates_straitjacket_n50.html) | Comparative TCO Report: `straitjacket` on `SWE-bench Pro (Candidates)` (N=44) |

**Gaps in the numbering.** Indices are never reused, so a missing number is a report that was withdrawn:

- **07** — SWE-bench Pro N=10 — withdrawn: the SWE-bench Pro path never ran the repository's tests, so its rows were canonical-patch substring scores, not pass rates.
- **08** — SWE-bench Pro N=30 — withdrawn for the same reason as 07.
- **09** — SWE-bench Pro N=30 (live API) — withdrawn for the same reason as 07.
- **14** — never produced a report file.
- **18** — a byte-identical duplicate of 19, written by the same sweep. Removed so the N=148 result has one address.

`inferred` dates: exact run timestamps were not recorded for those sweeps.
Their order is derived from task count and model generation, which is why it
is labelled rather than stated as fact.

Methodology documents live in [`../docs/`](../docs/), not here — this
directory holds run results only.
