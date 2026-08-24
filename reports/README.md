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

**Gaps in the numbering.** Indices are never reused, so a missing number is a report that was withdrawn:

- **07** — SWE-bench Pro N=10 — withdrawn: the SWE-bench Pro path never ran the repository's tests, so its rows were canonical-patch substring scores, not pass rates.
- **08** — SWE-bench Pro N=30 — withdrawn for the same reason as 07.
- **09** — SWE-bench Pro N=30 (live API) — withdrawn for the same reason as 07.
- **14** — never produced a report file.

`inferred` dates: exact run timestamps were not recorded for those sweeps.
Their order is derived from task count and model generation, which is why it
is labelled rather than stated as fact.

Methodology documents live in [`../docs/`](../docs/), not here — this
directory holds run results only.
